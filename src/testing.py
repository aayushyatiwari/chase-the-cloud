"""
h5explore.py -- deep structural inspection of HDF5 files.

Written for INSAT-3D/3DR Imager L1B/L1C products but works on any .h5.

Answers the questions that decide your preprocessing strategy:
  * what dtype is each dataset actually stored as?
  * is it chunked? compressed? with which filter and what real ratio?
  * what is the fill value, and how much of the array is fill?
  * can this dtype be narrowed losslessly?
  * which datasets are lookup tables rather than image data?

Everything is sampled. No dataset is ever read in full.

Usage:
    from h5explore import explore
    report = explore("3DIMG_15JUL2024_0000_L1C_ASIA_MER_V01R00.h5")

    explore("file.h5", verbose=False)          # quiet, just return dict
    explore("file.h5", sample_bytes=200e6)     # read bigger samples
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:
    import hdf5plugin  # noqa: F401  -- registers Blosc/Blosc2/Zstd/LZ4 filters
    _HAS_PLUGIN = True
except ImportError:
    _HAS_PLUGIN = False

import h5py

# ---------------------------------------------------------------------------
# HDF5 filter IDs -> human names. h5py only names the ones it ships with.
# ---------------------------------------------------------------------------
FILTER_NAMES = {
    1: "gzip/deflate",
    2: "shuffle",
    3: "fletcher32(checksum)",
    4: "szip",
    5: "nbit",
    6: "scaleoffset",
    307: "bzip2",
    32000: "lzf",
    32001: "blosc",
    32004: "lz4",
    32008: "bitshuffle",
    32013: "zfp",
    32015: "zstd",
    32026: "blosc2",
}

_UINT = [np.uint8, np.uint16, np.uint32]
_INT = [np.int8, np.int16, np.int32]


def _human(n: float) -> str:
    """Bytes -> readable string."""
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0


def _clean_attr(v: Any) -> Any:
    """Decode bytes, unwrap 1-element arrays, make JSON-safe."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip().strip("\x00")
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return _clean_attr(v.reshape(-1)[0])
        if v.size <= 12:
            return [_clean_attr(x) for x in v.reshape(-1)]
        return f"<array {v.shape} {v.dtype}>"
    if isinstance(v, np.generic):
        return v.item()
    return v


def _attrs(obj) -> dict:
    return {k: _clean_attr(v) for k, v in obj.attrs.items()}


def _filters(dset: h5py.Dataset) -> list[dict]:
    """
    Read the raw filter pipeline from the dataset creation property list.

    Necessary because dset.compression only reports gzip/lzf/szip; anything
    from hdf5plugin (blosc2, zstd, bitshuffle) shows up as None there even
    though the data is very much compressed.
    """
    out = []
    try:
        plist = dset.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            code, _flags, cd_values, name = plist.get_filter(i)[:4]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace").strip("\x00")
            out.append({
                "id": int(code),
                "name": FILTER_NAMES.get(int(code)) or (name or f"unknown({code})"),
                "client_data": [int(x) for x in cd_values],
            })
    except Exception as exc:  # pragma: no cover
        out.append({"error": repr(exc)})
    return out


def _sample(dset: h5py.Dataset, budget_bytes: float) -> np.ndarray | None:
    """
    Pull a strided sample under budget_bytes.

    Strides along every axis rather than taking a contiguous head, so we see
    the whole spatial field (edges, off-disk corners) instead of just the top
    few rows -- which for a full-disk image would be pure fill value and would
    make every statistic a lie.
    """
    if dset.size == 0:
        return None
    itemsize = max(dset.dtype.itemsize, 1)
    want = max(int(budget_bytes // itemsize), 1)

    if dset.size <= want:
        sl = tuple(slice(None) for _ in dset.shape)
    else:
        # spread the reduction factor across axes by its n-th root
        ndim = max(dset.ndim, 1)
        factor = (dset.size / want) ** (1.0 / ndim)
        step = max(int(np.ceil(factor)), 1)
        sl = tuple(slice(None, None, step if n > step else 1) for n in dset.shape)

    try:
        return np.asarray(dset[sl])
    except Exception:
        return None


def _narrowing(a: np.ndarray) -> dict:
    """
    Can this dtype be narrowed with zero loss? Tested on the sample, so the
    verdict is 'consistent with the sample', never a guarantee for the file.
    """
    res: dict[str, Any] = {}
    flat = a.reshape(-1)
    finite = flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
    if finite.size == 0:
        return {"note": "no finite values in sample"}

    res["min"] = float(finite.min())
    res["max"] = float(finite.max())
    res["n_unique_in_sample"] = int(np.unique(finite).size)

    if flat.dtype.kind == "f":
        res["nan_frac"] = float(np.mean(~np.isfinite(flat)))
        # is every value a whole number? => it was integer data all along
        integral = bool(np.all(finite == np.round(finite)))
        res["all_values_integral"] = integral
        if integral:
            lo, hi = finite.min(), finite.max()
            for dt in (_UINT if lo >= 0 else _INT):
                info = np.iinfo(dt)
                if lo >= info.min and hi <= info.max:
                    res["lossless_int_dtype"] = np.dtype(dt).name
                    res["int_saving_vs_current"] = f"{flat.dtype.itemsize / np.dtype(dt).itemsize:.1f}x"
                    break
        # float64 -> float32 round trip
        if flat.dtype == np.float64:
            rt = finite.astype(np.float32).astype(np.float64)
            res["float32_is_lossless"] = bool(np.array_equal(rt, finite))
            if not res["float32_is_lossless"]:
                denom = np.where(finite == 0, 1, np.abs(finite))
                res["float32_max_rel_err"] = float(np.max(np.abs(rt - finite) / denom))
    else:
        lo, hi = int(finite.min()), int(finite.max())
        for dt in (_UINT if lo >= 0 else _INT):
            info = np.iinfo(dt)
            if lo >= info.min and hi <= info.max:
                if np.dtype(dt).itemsize < flat.dtype.itemsize:
                    res["lossless_int_dtype"] = np.dtype(dt).name
                    res["int_saving_vs_current"] = f"{flat.dtype.itemsize / np.dtype(dt).itemsize:.1f}x"
                break
        # bits actually in use -- 10-bit sensor counts in a uint16 waste 6 bits
        if hi > 0:
            res["bits_used"] = int(hi).bit_length()
            res["bits_allocated"] = flat.dtype.itemsize * 8
    return res


def _fill_fraction(a: np.ndarray, fill) -> float | None:
    if fill is None:
        return None
    try:
        if isinstance(fill, float) and np.isnan(fill):
            return float(np.mean(np.isnan(a)))
        return float(np.mean(a == fill))
    except Exception:
        return None


def _looks_like_lut(dset: h5py.Dataset, dims: list[str]) -> bool:
    """
    LUT heuristic: 1-D, small, and either named like a calibration table or
    sitting on a GreyCount-style axis. For INSAT the tables are 1024 entries
    on a 'GreyCount' dimension, e.g. IMG_TIR1_TEMP.
    """
    if dset.ndim != 1 or dset.shape[0] > 65536:
        return False
    if any("greycount" in str(d).lower() or "count" == str(d).lower() for d in dims):
        return True
    name = dset.name.upper()
    return any(k in name for k in ("_TEMP", "_RADIANCE", "_ALBEDO", "LUT", "_BT"))


def _dim_labels(dset: h5py.Dataset) -> list[str]:
    labels = []
    try:
        for i in range(dset.ndim):
            d = dset.dims[i]
            if d.label:
                labels.append(d.label)
            elif len(d):
                labels.append(os.path.basename(d[0].name))
            else:
                labels.append("")
    except Exception:
        labels = [""] * dset.ndim
    return labels


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def explore(
    path: str,
    sample_bytes: float = 50e6,
    verbose: bool = True,
    max_depth: int | None = None,
    stats: bool = True,
) -> dict:
    """
    Inspect an HDF5 file and return a structured report.

    Parameters
    ----------
    path         : file to inspect.
    sample_bytes : approximate read budget per dataset. Raise for tighter
                   statistics, lower for a fast structural pass.
    verbose      : print a human-readable report as well as returning it.
    max_depth    : stop descending below this group depth (None = no limit).
    stats        : compute value statistics and narrowing checks. Set False
                   for a pure structure dump with no data reads.

    Returns
    -------
    dict with keys: file, root_attrs, groups, datasets, softlinks,
    external_links, totals, findings.
    """
    if not _HAS_PLUGIN:
        print("! hdf5plugin not installed -- blosc/zstd/bitshuffle datasets "
              "will fail to read. pip install hdf5plugin")

    report: dict[str, Any] = {
        "file": {
            "path": os.path.abspath(path),
            "size_on_disk": os.path.getsize(path),
            "size_on_disk_h": _human(os.path.getsize(path)),
        },
        "root_attrs": {},
        "groups": [],
        "datasets": {},
        "softlinks": {},
        "external_links": {},
        "totals": {},
        "findings": [],
    }

    with h5py.File(path, "r") as f:
        report["file"]["libver"] = str(f.libver)
        report["file"]["userblock"] = int(f.userblock_size)
        report["root_attrs"] = _attrs(f)

        # -- catch links before h5py silently resolves or drops them --------
        def scan_links(group, depth=0):
            for key in group:
                try:
                    link = group.get(key, getlink=True)
                except Exception:
                    continue
                full = f"{group.name.rstrip('/')}/{key}"
                if isinstance(link, h5py.SoftLink):
                    report["softlinks"][full] = link.path
                elif isinstance(link, h5py.ExternalLink):
                    report["external_links"][full] = f"{link.filename}::{link.path}"
                else:
                    obj = group.get(key)
                    if isinstance(obj, h5py.Group) and (max_depth is None or depth < max_depth):
                        scan_links(obj, depth + 1)

        scan_links(f)

        raw_total = 0
        stored_total = 0

        def visit(name, obj):
            nonlocal raw_total, stored_total
            depth = name.count("/")
            if max_depth is not None and depth > max_depth:
                return

            if isinstance(obj, h5py.Group):
                report["groups"].append({
                    "path": "/" + name,
                    "n_members": len(obj),
                    "attrs": _attrs(obj),
                })
                return

            if not isinstance(obj, h5py.Dataset):
                return

            raw = int(obj.size) * int(obj.dtype.itemsize)
            try:
                stored = int(obj.id.get_storage_size())
            except Exception:
                stored = None
            raw_total += raw
            if stored:
                stored_total += stored

            dims = _dim_labels(obj)
            info: dict[str, Any] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "dim_labels": dims,
                "n_elements": int(obj.size),
                "raw_bytes": raw,
                "raw_h": _human(raw),
                "stored_bytes": stored,
                "stored_h": _human(stored),
                "ratio": round(raw / stored, 2) if stored else None,
                "layout": ("chunked" if obj.chunks
                           else "compact" if raw and stored and stored < 64 * 1024 and obj.chunks is None and raw <= 64 * 1024
                           else "contiguous"),
                "chunks": list(obj.chunks) if obj.chunks else None,
                "chunk_bytes": (int(np.prod(obj.chunks)) * obj.dtype.itemsize
                                if obj.chunks else None),
                "filters": _filters(obj),
                "fillvalue": _clean_attr(obj.fillvalue) if obj.shape else None,
                "scaleoffset": obj.scaleoffset,
                "attrs": _attrs(obj),
                "is_lut": _looks_like_lut(obj, dims),
            }
            if info["chunk_bytes"]:
                info["chunk_h"] = _human(info["chunk_bytes"])

            # _FillValue attribute takes precedence over the HDF5 fillvalue
            declared_fill = info["attrs"].get("_FillValue", info["fillvalue"])
            info["declared_fill"] = declared_fill

            if stats and obj.size:
                a = _sample(obj, sample_bytes)
                if a is not None and a.size:
                    info["sample_shape"] = list(a.shape)
                    info["sample_frac"] = round(a.size / obj.size, 4)
                    info["fill_frac_in_sample"] = _fill_fraction(a, declared_fill)
                    try:
                        info["narrowing"] = _narrowing(a)
                    except Exception as exc:
                        info["narrowing"] = {"error": repr(exc)}
                    if a.dtype.kind in "fiu":
                        valid = a
                        if declared_fill is not None and a.dtype.kind in "iu":
                            valid = a[a != declared_fill]
                        if valid.size:
                            info["stats"] = {
                                "mean": round(float(np.mean(valid)), 4),
                                "std": round(float(np.std(valid)), 4),
                                "p1": round(float(np.percentile(valid, 1)), 4),
                                "p50": round(float(np.percentile(valid, 50)), 4),
                                "p99": round(float(np.percentile(valid, 99)), 4),
                            }
                        if info["is_lut"] and a.size <= 8:
                            info["lut_preview"] = [float(x) for x in a.reshape(-1)]
                        elif info["is_lut"]:
                            flat = a.reshape(-1)
                            info["lut_endpoints"] = [float(flat[0]), float(flat[-1])]
                            d = np.diff(flat.astype(np.float64))
                            nz = d[d != 0]
                            info["lut_monotonic"] = bool(np.all(nz > 0) or np.all(nz < 0))
                            info["lut_linear"] = bool(
                                nz.size and np.allclose(nz, nz[0], rtol=1e-6)
                            )

            report["datasets"]["/" + name] = info

        f.visititems(visit)

        report["totals"] = {
            "n_groups": len(report["groups"]),
            "n_datasets": len(report["datasets"]),
            "raw_bytes": raw_total,
            "raw_h": _human(raw_total),
            "stored_bytes": stored_total,
            "stored_h": _human(stored_total),
            "overall_ratio": round(raw_total / stored_total, 2) if stored_total else None,
        }

    report["findings"] = _findings(report)

    if verbose:
        _print(report)
    return report


# ---------------------------------------------------------------------------
def _findings(report: dict) -> list[str]:
    """Turn the raw numbers into things worth acting on."""
    out = []
    for path, d in report["datasets"].items():
        if d["n_elements"] < 1024 or d.get("is_lut"):
            continue
        n = _human(d["raw_bytes"])

        if not d["filters"]:
            out.append(f"{path}: uncompressed ({n} raw) -- add blosc2+zstd+bitshuffle")
        if d["layout"] == "contiguous" and d["raw_bytes"] > 1e6:
            out.append(f"{path}: contiguous, not chunked -- no partial reads, no compression possible")

        nar = d.get("narrowing", {})
        if nar.get("lossless_int_dtype"):
            out.append(f"{path}: {d['dtype']} -> {nar['lossless_int_dtype']} "
                       f"is lossless on the sample ({nar.get('int_saving_vs_current')} smaller)")
        if nar.get("float32_is_lossless"):
            out.append(f"{path}: float64 -> float32 is lossless on the sample (2.0x smaller)")
        if nar.get("all_values_integral") and d["dtype"].startswith("float"):
            out.append(f"{path}: stored as {d['dtype']} but every sampled value is a whole "
                       f"number ({nar.get('n_unique_in_sample')} unique) -- likely decoded "
                       f"sensor counts, store the counts instead")

        # The LUT-decoded signature: a float array whose value set is tiny
        # relative to its pixel count. Carries at most log2(k) bits per pixel
        # but pays 32 or 64. This is what float32 brightness temperature from a
        # 1024-entry calibration table looks like.
        k = nar.get("n_unique_in_sample")
        n_s = int(d["n_elements"] * d.get("sample_frac", 1.0))
        if (d["dtype"].startswith("float") and k and n_s > 20000
                and k <= 4096 and k < n_s / 20):
            need = max(int(k - 1).bit_length(), 1)
            idx = "uint8" if k <= 256 else "uint16"
            out.append(
                f"{path}: only {k} distinct values across {n_s:,} sampled pixels "
                f"-- {need} bits of real information stored in {d['dtype']}. "
                f"This is decoded lookup-table output. Store the original "
                f"{idx} index/count + the table ({k} values) instead: "
                f"{d['dtype'].replace('float','')} -> {idx} is "
                f"{np.dtype(d['dtype']).itemsize / np.dtype(idx).itemsize:.0f}x smaller "
                f"and bit-exact.")
        # blosc/blosc2 apply shuffle+bitshuffle internally via client data, so
        # only recommend it when no blosc-family filter is in the pipeline.
        fnames = {x.get("name", "") for x in d["filters"]}
        has_shuffle = any("blosc" in n or "shuffle" in n for n in fnames)
        if (nar.get("bits_used") and not has_shuffle
                and nar["bits_used"] <= nar.get("bits_allocated", 99) - 4):
            out.append(f"{path}: only {nar['bits_used']} of {nar['bits_allocated']} bits used "
                       f"-- bitshuffle will help a lot")

        ff = d.get("fill_frac_in_sample")
        if ff and ff > 0.15:
            out.append(f"{path}: {ff:.0%} of sample is fill value {d['declared_fill']} "
                       f"-- compresses to nearly nothing, and exclude it from normalisation")

        cb = d.get("chunk_bytes")
        if cb and cb < 64 * 1024:
            out.append(f"{path}: chunks are only {_human(cb)} -- too small, aim for 1-16MB")
        if cb and cb > 64e6:
            out.append(f"{path}: chunks are {_human(cb)} -- large, every read decompresses all of it")
        if d["chunks"] and len(d["chunks"]) == 3 and d["chunks"][0] == 1 and d["shape"][0] > 1:
            out.append(f"{path}: chunked with time=1 -- a temporal stack of N frames costs N "
                       f"chunk reads; chunk across time for nowcasting")
    if report["external_links"]:
        out.append(f"{len(report['external_links'])} external link(s) -- this file is not "
                   f"self-contained, copying it alone will break it")
    return out


def _print(report: dict) -> None:
    W = 78
    print("=" * W)
    print(f"FILE  {report['file']['path']}")
    print(f"      {report['file']['size_on_disk_h']} on disk")
    t = report["totals"]
    print(f"      {t['n_datasets']} datasets, {t['n_groups']} groups | "
          f"raw {t['raw_h']} -> stored {t['stored_h']}"
          + (f" ({t['overall_ratio']}x)" if t["overall_ratio"] else ""))
    print("=" * W)

    if report["root_attrs"]:
        print("\nROOT ATTRIBUTES")
        for k, v in report["root_attrs"].items():
            print(f"  {k:<34} {str(v)[:40]}")

    luts = {k: v for k, v in report["datasets"].items() if v.get("is_lut")}
    main = {k: v for k, v in report["datasets"].items() if not v.get("is_lut")}

    if main:
        print("\nDATASETS")
        for path, d in sorted(main.items(), key=lambda kv: -kv[1]["raw_bytes"]):
            print(f"\n  {path}")
            dims = ",".join(x or "?" for x in d["dim_labels"])
            print(f"    shape {tuple(d['shape'])} [{dims}]  dtype {d['dtype']}")
            line = f"    raw {d['raw_h']} -> stored {d['stored_h']}"
            if d["ratio"]:
                line += f"  ({d['ratio']}x)"
            print(line)
            print(f"    layout {d['layout']}"
                  + (f"  chunks {tuple(d['chunks'])} = {d.get('chunk_h')}" if d["chunks"] else ""))
            if d["filters"]:
                print(f"    filters {' -> '.join(x.get('name', '?') for x in d['filters'])}")
            else:
                print("    filters (none)")
            if d["declared_fill"] is not None:
                ff = d.get("fill_frac_in_sample")
                print(f"    fill {d['declared_fill']}"
                      + (f"  ({ff:.1%} of sample)" if ff is not None else ""))
            if "stats" in d:
                s = d["stats"]
                print(f"    valid p1/p50/p99 {s['p1']} / {s['p50']} / {s['p99']}"
                      f"   mean {s['mean']} sd {s['std']}")
            nar = d.get("narrowing", {})
            if nar.get("n_unique_in_sample") is not None:
                bits = (f", {nar['bits_used']}/{nar['bits_allocated']} bits used"
                        if nar.get("bits_used") else "")
                print(f"    {nar['n_unique_in_sample']} unique in sample "
                      f"(range {nar.get('min')}..{nar.get('max')}){bits}")
            skip = {"units", "long_name", "_FillValue"}
            extra = {k: v for k, v in d["attrs"].items() if k not in skip}
            for k in ("units", "long_name"):
                if k in d["attrs"]:
                    print(f"    {k}: {d['attrs'][k]}")
            if extra:
                print(f"    attrs: {json.dumps(extra, default=str)[:150]}")

    if luts:
        print("\nLOOKUP TABLES  (carry these -- calibration is not a formula)")
        for path, d in sorted(luts.items()):
            bits = []
            if "lut_endpoints" in d:
                lo, hi = d["lut_endpoints"]
                bits.append(f"{lo:g}..{hi:g}")
            if d.get("lut_monotonic") is not None:
                bits.append("monotonic" if d["lut_monotonic"] else "NON-monotonic")
            if d.get("lut_linear") is not None:
                bits.append("LINEAR (gain/offset would do)" if d["lut_linear"] else "non-linear")
            u = d["attrs"].get("units", "")
            print(f"  {path:<34} n={d['shape'][0]:<6} {u:<8} {'  '.join(bits)}")

    if report["softlinks"] or report["external_links"]:
        print("\nLINKS")
        for k, v in report["softlinks"].items():
            print(f"  soft      {k} -> {v}")
        for k, v in report["external_links"].items():
            print(f"  external  {k} -> {v}")

    if report["findings"]:
        print("\nFINDINGS")
        seen = set()
        for msg in report["findings"]:
            if msg not in seen:
                print(f"  * {msg}")
                seen.add(msg)
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python h5explore.py FILE.h5 [sample_MB]")
    mb = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    explore(sys.argv[1], sample_bytes=mb * 1e6)
