"""Notebook display helpers: symlinks, HTML grids, clickable images."""
from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path

from IPython.display import HTML, display
from PIL import Image


# ----- Stage 0 display -------------------------------------------------------

def plot_exposure_group_thumbnails(exposure_groups: dict, thumb_width: int = 150) -> None:
    """Show one thumbnail per exposure group, sorted by exposure time."""
    cells = []
    for exp_time, infos in sorted(exposure_groups.items()):
        if not infos:
            continue
        img = Image.open(infos[0].path)
        thumb_height = thumb_width * img.height // img.width
        img.thumbnail((thumb_width, thumb_height))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        cap = html.escape(f"{exp_time:.5f}s ({len(infos)} imgs)")
        cells.append(
            f'<td style="vertical-align:top; text-align:center; padding:6px;">'
            f'<img src="data:image/jpeg;base64,{b64}" style="width:{thumb_width}px;">'
            f'<p style="margin:4px 0 0 0;"><small>{cap}</small></p>'
            f'</td>'
        )
    display(HTML(
        '<table style="border-collapse:collapse;"><tr>'
        + "".join(cells)
        + "</tr></table>"
    ))


# ----- Filename patterns ------------------------------------------------------

_STAGE1_DEBUG_ANIM_STRICT = re.compile(r"^v6-stage1_debugimg_(.+)_anim\.gif$")
_STAGE2_PAIR_STRICT = re.compile(
    r'^v6-stage2_pair_(\d+\.\d+)_(\d+\.\d+)_gamma(\d+\.\d+)\.gif$'
)


# ----- Shared helper ----------------------------------------------------------

def _rel_href_for_notebook(path: Path) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        rel = path.relative_to(cwd)
    except ValueError:
        rel = Path(path.name)
    s = rel.as_posix()
    if not s.startswith(("./", "/")):
        s = "./" + s
    return s


# ----- Stage 1 display -------------------------------------------------------

def symlink_stage1_debug_anim_gifs(workdir: Path, link_dir: Path | None = None) -> list[Path]:
    """
    For each strict-match `v6-stage1_debugimg_*_anim.gif` under `workdir`, create a symlink in
    `link_dir` (default: cwd) with the same basename, pointing at the resolved source file.
    Removes an existing file or symlink at the destination before creating the link.
    """
    workdir = Path(workdir)
    link_dir = Path.cwd() if link_dir is None else Path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for src in sorted(workdir.glob("v6-stage1_debugimg*_anim.gif"), key=lambda p: p.name):
        if not _STAGE1_DEBUG_ANIM_STRICT.match(src.name):
            continue
        if not src.is_file():
            continue
        dst = link_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        created.append(dst)
    return created


def display_clickable_stage1_debug_img_grid(columns: int = 8, width: int = 128, gif_dir: Path | None = None) -> None:
    """
    Show strict-match `v6-stage1_debugimg_*_anim.gif` under `gif_dir` (default: cwd) in a table
    with `columns` columns, lexicographic order by filename. Each cell is a clickable thumbnail
    like `display_clickable_img`, with caption ``Exposure <substring> s`` from the filename.
    """
    gif_dir = Path.cwd() if gif_dir is None else Path(gif_dir)
    chunk: list[tuple[Path, str]] = []
    for p in sorted(gif_dir.glob("v6-stage1_debugimg*_anim.gif"), key=lambda q: q.name):
        m = _STAGE1_DEBUG_ANIM_STRICT.match(p.name)
        if not m:
            continue
        chunk.append((p, m.group(1)))
    if not chunk:
        display(HTML("<p><em>No strict-match v6-stage1_debugimg_*_anim.gif files found.</em></p>"))
        return
    rows_html: list[str] = []
    for i in range(0, len(chunk), columns):
        row_cells: list[str] = []
        for p, exposure in chunk[i : i + columns]:
            href = html.escape(_rel_href_for_notebook(p), quote=True)
            cap = html.escape(f"Exposure {exposure} s", quote=False)
            row_cells.append(
                "<td style=\"vertical-align:top; text-align:center; padding:6px;\">"
                f'<a href="{href}" target="_blank">'
                f'<img src="{href}" style="width:{int(width)}px; border:1px solid #ccc; border-radius:5px;">'
                "</a>"
                f'<p style="margin:4px 0 0 0;"><small>{cap}</small></p>'
                "</td>"
            )
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
    display(HTML(
        '<table style="border-collapse:collapse;">'
        + "".join(rows_html)
        + "</table>"
    ))


# ----- Stage 2 display -------------------------------------------------------

def symlink_stage2_pair_gifs(workdir: Path, link_dir: Path | None = None) -> list[Path]:
    workdir = Path(workdir)
    link_dir = Path.cwd() if link_dir is None else Path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for src in sorted(workdir.glob("v6-stage2_pair*.gif"), key=lambda p: p.name):
        if not _STAGE2_PAIR_STRICT.match(src.name):
            continue
        if not src.is_file():
            continue
        dst = link_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        created.append(dst)
    return created


def display_clickable_stage2_pair_gif_grid(
    columns: int = 8, width: int = 128, gif_dir: Path | None = None
) -> None:
    gif_dir = Path.cwd() if gif_dir is None else Path(gif_dir)
    chunk: list[tuple[Path, str]] = []
    for p in sorted(gif_dir.glob("v6-stage2_pair*.gif"), key=lambda q: q.name):
        m = _STAGE2_PAIR_STRICT.match(p.name)
        if not m:
            continue
        t0, t1, gamma = m.group(1), m.group(2), m.group(3)
        chunk.append((p, f"t0={t0} t1={t1} \u03b3={gamma}"))
    if not chunk:
        display(HTML("<p><em>No strict-match v6-stage2_pair_*.gif files found.</em></p>"))
        return
    rows_html: list[str] = []
    for i in range(0, len(chunk), columns):
        row_cells: list[str] = []
        for p, cap in chunk[i : i + columns]:
            href = html.escape(_rel_href_for_notebook(p), quote=True)
            cap_esc = html.escape(cap, quote=False)
            row_cells.append(
                "<td style=\"vertical-align:top; text-align:center; padding:6px;\">"
                f'<a href="{href}" target="_blank">'
                f'<img src="{href}" style="width:{int(width)}px; border:1px solid #ccc; border-radius:5px;">'
                "</a>"
                f'<p style="margin:4px 0 0 0;"><small>{cap_esc}</small></p>'
                "</td>"
            )
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
    display(HTML(
        '<table style="border-collapse:collapse;">'
        + "".join(rows_html)
        + "</table>"
    ))


# ----- Stage 3 display -------------------------------------------------------

def symlink_and_display_clickable(ctx, filename: str) -> None:
    src = ctx.workdir / filename
    dst = Path.cwd() / filename
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    href = html.escape(_rel_href_for_notebook(dst), quote=True)
    display(
        HTML(
            f'<a href="{href}" target="_blank">'
            f'<img src="{href}" style="width:256px; border:1px solid #ccc; border-radius:5px;">'
            "</a>"
        )
    )
