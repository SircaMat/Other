# streamlit_pdf_trimmer.py
# Streamlit app to upload one or more PDFs, pick pages to keep/remove per file,
# preview inputs and outputs, and download optimized (smaller) PDFs.
# Includes: top instructions, per-field tooltips, linearization-safe saving,
# token consumption estimation, and saving outputs to app folder as <name>_trimmed.pdf.

import io
import os
import math
from typing import List, Set, Tuple, Optional

import fitz  # PyMuPDF
import streamlit as st

# Try optional tiktoken for more accurate token counting
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# ---------------- Utilities ----------------

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(size)} {u}"
            return f"{size:.2f} {u}"
        size /= 1024

def parse_page_spec(spec: str, page_count: int) -> List[int]:
    """
    Parse a 1-based page list like '1,3,5-7' -> sorted 0-based unique indices.
    Ignores out-of-range entries; raises ValueError on malformed tokens.
    """
    if not spec:
        return []
    result: Set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                raise ValueError(f"Invalid range '{part}'")
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                if 1 <= p <= page_count:
                    result.add(p - 1)
        else:
            try:
                p = int(part)
            except ValueError:
                raise ValueError(f"Invalid page '{part}'")
            if 1 <= p <= page_count:
                result.add(p - 1)
    return sorted(result)

def render_page_image(doc: fitz.Document, page_index: int, dpi: int = 150) -> bytes:
    """Render a single page to PNG bytes for preview."""
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

@st.cache_resource
def linearization_supported() -> bool:
    """
    Detect whether MuPDF in this environment supports PDF linearization.
    Tries once per session by saving a 1-page blank doc with linear=True.
    """
    try:
        test = fitz.open()
        test.new_page()
        buf = io.BytesIO()
        test.save(buf, linear=True)
        test.close()
        return True
    except Exception:
        return False

def extract_text_from_pdf_bytes(pdf_bytes: bytes, page_indices: Optional[List[int]] = None, chars_cap: Optional[int] = None) -> str:
    """
    Extract plain text from specified pages. If page_indices is None, extract all pages.
    chars_cap: optional cap to avoid extremely large text concatenations.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        indices = page_indices if page_indices is not None else list(range(doc.page_count))
        parts: List[str] = []
        total_len = 0
        for i in indices:
            if 0 <= i < doc.page_count:
                try:
                    txt = (doc.load_page(i).get_text("text") or "")
                except Exception:
                    txt = ""
                parts.append(txt)
                total_len += len(txt)
                if chars_cap is not None and total_len >= chars_cap:
                    break
        text = "\n".join(parts)
        if chars_cap is not None and len(text) > chars_cap:
            text = text[:chars_cap]
        return text
    finally:
        doc.close()

def count_tokens_from_text(
    text: str,
    method: str = "auto",
    encoding_name: str = "o200k_base",
) -> int:
    """
    Return an estimated token count for a given text:
    - method="auto" -> use tiktoken if available with specified encoding_name, else fallback to chars/4
    - method="approx_chars" -> ceil(len(text)/4)
    - method="approx_words" -> ceil(words * 1.3)
    """
    if not text:
        return 0

    if method == "approx_chars":
        return int(math.ceil(len(text) / 4.0))
    if method == "approx_words":
        words = len(text.split())
        return int(math.ceil(words * 1.3))

    # auto
    if _HAS_TIKTOKEN:
        enc_name = encoding_name or "o200k_base"
        try:
            enc = tiktoken.get_encoding(enc_name)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return int(math.ceil(len(text) / 4.0))
        try:
            return len(enc.encode(text))
        except Exception:
            return int(math.ceil(len(text) / 4.0))
    else:
        return int(math.ceil(len(text) / 4.0))

def optimize_and_trim(
    pdf_bytes: bytes,
    keep_indices: Optional[List[int]] = None,
    remove_indices: Optional[List[int]] = None,
    garbage: int = 3,
    deflate: bool = True,
    clean: bool = True,
    linear: bool = False,   # default False to avoid errors on recent MuPDF
) -> Tuple[bytes, int, int]:
    """
    Returns (output_bytes, in_pages, out_pages)
    - Applies keep/remove logic.
    - Saves with lossless optimizations.
    - If linearization requested but unsupported, falls back safely.
    """
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    in_pages = src.page_count

    # Apply keep/remove
    if keep_indices:
        new_doc = fitz.open()
        for i in keep_indices:
            if 0 <= i < src.page_count:
                new_doc.insert_pdf(src, from_page=i, to_page=i)
        src.close()
        doc = new_doc
    elif remove_indices:
        doc = src
        for i in sorted(set(remove_indices), reverse=True):
            if 0 <= i < doc.page_count:
                doc.delete_page(i)
    else:
        doc = src

    out_buf = io.BytesIO()
    save_kwargs = dict(garbage=garbage, deflate=deflate, clean=clean)

    if linear and linearization_supported():
        try:
            doc.save(out_buf, linear=True, **save_kwargs)
        except Exception:
            out_buf = io.BytesIO()
            doc.save(out_buf, **save_kwargs)
    else:
        doc.save(out_buf, **save_kwargs)

    doc.close()

    out_bytes = out_buf.getvalue()
    out_pages = fitz.open(stream=out_bytes, filetype="pdf").page_count
    return out_bytes, in_pages, out_pages

def unique_path_in_dir(directory: str, filename: str) -> str:
    """
    Return a unique path inside 'directory' for 'filename'.
    If the exact file exists, append ' (1)', ' (2)', etc. before extension.
    """
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    k = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base} ({k}){ext}")
        k += 1
    return candidate

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="PDF Trimmer & Optimizer", page_icon="📄", layout="wide")

st.title("📄 PDF Trimmer & Optimizer (Streamlit)")

# Top instructions dropdown
with st.expander("❓ How to use this app (click to expand)"):
    st.markdown(
        """
**Goal:** Reduce PDF size and token usage by keeping only relevant pages or removing unnecessary ones.

**Steps:**
1. **Upload** one or more PDF files below.
2. For **each file**:
   - Select **Mode**:
     - **Keep** – keep only the pages you specify.
     - **Remove** – remove the pages you specify, keep the rest.
     - **Just optimize** – no page changes, just compact the PDF.
   - (Optional) Enter **Password** if the PDF is encrypted.
   - Enter **Pages** in the format `1,3,5-7` (1-based, ranges inclusive).
   - Use **Preview controls** to see page images and text snippets (helps page selection).
3. Click **Process this PDF** to generate the trimmed & optimized file.
4. Review the **Output preview** and **download** the result.

**Token estimates:** We extract text and estimate tokens. Scanned/image-only pages may show very low token counts (vision/image tokens are **not** included).
"""
    )

st.markdown(
    """
Upload one or more **PDFs**, choose to **KEEP** specific pages or **REMOVE** unwanted ones,
then download **smaller, optimized** PDFs. Previews help you verify selections.
"""
)

# Sidebar with tooltips & optional popovers
with st.sidebar:
    st.header("⚙️ Options")

    # Optional popover (auto-skipped if not supported by your Streamlit version)
    try:
        with st.popover("ℹ️ About previews"):
            st.write(
                "Rendering high-DPI previews or many pages can be slower. "
                "Use the sliders to balance quality and speed."
            )
    except Exception:
        pass

    preview_dpi = st.slider(
        "Preview DPI",
        min_value=72, max_value=220, value=140, step=4,
        help="Resolution for page image previews only (does not affect PDF output). Higher = sharper but slower."
    )
    text_snippet = st.slider(
        "Text snippet length (chars)",
        80, 600, 220, 20,
        help="How many characters to show per page in the quick text preview."
    )
    max_preview_pages = st.slider(
        "Max preview pages to render",
        1, 12, 4, 1,
        help="Upper limit of how many pages are rendered as images at once."
    )
    st.caption("Higher DPI or more pages = slower previews.")

    st.subheader("Save / Optimization")
    supports_linear = linearization_supported()
    opt_garbage = st.select_slider(
        "Garbage collection depth",
        options=[0, 1, 2, 3], value=3,
        help="Removes unreferenced objects. 3 = deepest cleanup (recommended)."
    )
    opt_deflate = st.checkbox(
        "Deflate content streams", value=True,
        help="Compresses PDF streams where possible (lossless)."
    )
    opt_clean = st.checkbox(
        "Clean & normalize", value=True,
        help="Fixes minor structural issues and normalizes objects (lossless)."
    )
    opt_linear = st.checkbox(
        "Linearize for web (fast first page)", value=False,
        help="May not be supported by your MuPDF build. If unsupported, the app will fall back automatically.",
        disabled=not supports_linear,
    )

    st.subheader("Token estimation")
    token_method = st.selectbox(
        "Method",
        options=["auto", "approx_chars", "approx_words"],
        format_func=lambda m: {
            "auto": "Auto (use tiktoken if available, else fallback)",
            "approx_chars": "Approximate (characters ÷ 4)",
            "approx_words": "Approximate (words × 1.3)",
        }[m],
        help="Choose how to estimate tokens for LLM usage."
    )

    if token_method == "auto" and _HAS_TIKTOKEN:
        encoding_name = st.selectbox(
            "Encoding (for tiktoken)",
            options=["o200k_base", "cl100k_base"],
            index=0,
            help="o200k_base: GPT‑4o family; cl100k_base: GPT‑4 / GPT‑3.5 text models."
        )
    else:
        encoding_name = "o200k_base"  # not used in approximations

st.divider()

uploads = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drag & drop or browse to select PDFs. Multiple files are supported."
)

if not uploads:
    st.info("Upload one or more PDFs to begin.")
    st.stop()

# Keep per-file settings in session_state
if "file_settings" not in st.session_state:
    st.session_state.file_settings = {}

# Directory where this app file lives
APP_DIR = os.path.dirname(os.path.abspath(__file__))

for up in uploads:
    file_key = f"cfg::{up.name}::{up.size}"
    if file_key not in st.session_state.file_settings:
        st.session_state.file_settings[file_key] = {
            "mode": "keep",       # keep | remove | optimize
            "pages": "",
            "pwd": "",
            "preview_range": (1, 1),
            "show_preview": False,
        }

    cfg = st.session_state.file_settings[file_key]

    with st.expander(f"📁 {up.name}", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            cfg["pwd"] = st.text_input(
                "Password (if encrypted)",
                value=cfg.get("pwd", ""),
                type="password",
                key=f"pwd_{file_key}",
                help="If the PDF is password-protected, enter it here to enable preview and processing."
            )
        with col2:
            cfg["mode"] = st.selectbox(
                "Mode",
                options=["keep", "remove", "optimize"],
                format_func=lambda x: {
                    "keep": "Keep selected pages",
                    "remove": "Remove selected pages",
                    "optimize": "Just optimize",
                }[x],
                index=["keep", "remove", "optimize"].index(cfg.get("mode", "keep")),
                key=f"mode_{file_key}",
                help=(
                    "Keep: Only the pages you list are kept.\n"
                    "Remove: The pages you list are removed (others remain).\n"
                    "Just optimize: No page changes; compact file losslessly."
                )
            )
        with col3:
            cfg["pages"] = st.text_input(
                "Pages (1-based: 1,3,5-7)",
                value=cfg.get("pages", ""),
                key=f"pages_{file_key}",
                help="Enter pages using 1-based numbers and ranges. Examples: 3, 5-9, 12, 14-16."
            )

        # Load doc to show details & previews
        pdf_bytes = up.getvalue()
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            st.error(f"Failed to open PDF: {e}")
            continue

        if doc.needs_pass:
            if cfg["pwd"]:
                ok = doc.authenticate(cfg["pwd"])
                if not ok:
                    st.error("Password incorrect. Enter the correct password to proceed.")
                    continue
            else:
                st.warning("This PDF is encrypted. Provide password to preview or process.")
                continue
        total_pages = doc.page_count
        st.caption(f"Pages: {total_pages}")

        # Preview controls
        start_default = 1
        end_default = min(total_pages, max_preview_pages)
        cfg["preview_range"] = st.slider(
            "Preview page range (1-based)",
            1, total_pages,
            value=(
                cfg.get("preview_range", (start_default, end_default))[0],
                cfg.get("preview_range", (start_default, end_default))[1]
            ),
            key=f"pr_{file_key}",
            help="Select which pages to preview below. This does not affect the final output."
        )

        # Optional small popover near preview controls (if supported)
        try:
            with st.popover("ℹ️ Preview tips"):
                st.write(
                    "Use this to quickly inspect content before deciding which pages to keep or remove. "
                    "The preview range is capped by 'Max preview pages' in the sidebar."
                )
        except Exception:
            pass

        cfg["show_preview"] = st.checkbox(
            "Show page previews (images)",
            value=cfg.get("show_preview", False),
            key=f"sp_{file_key}",
            help="Toggle image previews for the selected page range."
        )

        if cfg["show_preview"]:
            s, e = cfg["preview_range"]
            e = min(e, s + max_preview_pages - 1)
            st.write(f"Rendering pages {s}–{e} (max {max_preview_pages})…")
            pcols = st.columns(min(3, e - s + 1))
            for idx, p in enumerate(range(s, e + 1)):
                with pcols[idx % len(pcols)]:
                    try:
                        img = render_page_image(doc, p - 1, dpi=preview_dpi)
                        st.image(img, caption=f"Page {p}", use_container_width=True)
                    except Exception as re:
                        st.warning(f"Preview failed for page {p}: {re}")

        # Quick text preview
        with st.expander("📝 Quick text preview (first ~N chars per page)"):
            st.caption("Useful for scanned vs. text PDFs and quick content checks.")
            s, e = cfg["preview_range"]
            e = min(e, s + max_preview_pages - 1)
            for p in range(s, e + 1):
                try:
                    txt = (doc.load_page(p - 1).get_text("text") or "").strip()
                except Exception:
                    txt = ""
                snip = (txt[:text_snippet] + ("…" if len(txt) > text_snippet else "")) if txt else "(no extractable text)"
                st.markdown(f"**Page {p}:** {snip}")

        # Validate page selection early
        keep_indices = remove_indices = None
        if cfg["mode"] in ("keep", "remove"):
            try:
                indices = parse_page_spec(cfg["pages"], total_pages)
                if not indices:
                    st.info("Enter at least one valid page in the 'Pages' field.")
                else:
                    if cfg["mode"] == "keep":
                        keep_indices = indices
                    else:
                        remove_indices = indices
            except ValueError as ve:
                st.error(str(ve))

        # Process button per file
        if st.button("Process this PDF", key=f"proc_{file_key}", help="Apply your selection and optimizations to this PDF."):
            if cfg["mode"] in ("keep", "remove") and not (keep_indices or remove_indices):
                st.error("Please provide a valid page list.")
            else:
                try:
                    out_bytes, in_pages, out_pages = optimize_and_trim(
                        pdf_bytes,
                        keep_indices=keep_indices,
                        remove_indices=remove_indices,
                        garbage=opt_garbage,
                        deflate=opt_deflate,
                        clean=opt_clean,
                        linear=opt_linear,
                    )
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                else:
                    in_size = len(pdf_bytes)
                    out_size = len(out_bytes)
                    saved = in_size - out_size
                    pct = (saved / in_size * 100) if in_size > 0 else 0

                    st.success(
                        f"Done! Pages: {in_pages} → {out_pages}. "
                        f"Size: {human_size(in_size)} → {human_size(out_size)} "
                        f"(saved {human_size(max(saved,0))}, {pct:.1f}%)."
                    )

                    # === Token estimation ===
                    st.subheader("🔢 Token estimate")
                    st.caption(
                        "We extract text from the PDFs and estimate tokens. "
                        "For scanned/image-only pages, text may be minimal—vision/image tokens are not included."
                    )

                    # Extract text for full original (all pages)
                    try:
                        original_text = extract_text_from_pdf_bytes(pdf_bytes)
                    except Exception:
                        original_text = ""

                    # Extract text for processed (open out_bytes and get all pages)
                    try:
                        processed_text = extract_text_from_pdf_bytes(out_bytes)
                    except Exception:
                        processed_text = ""

                    original_tokens = count_tokens_from_text(original_text, method=token_method, encoding_name=encoding_name)
                    processed_tokens = count_tokens_from_text(processed_text, method=token_method, encoding_name=encoding_name)
                    saved_tokens = max(0, original_tokens - processed_tokens)
                    pct_tokens = (saved_tokens / original_tokens * 100) if original_tokens > 0 else 0.0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Original tokens (est.)", f"{original_tokens:,}")
                    c2.metric("Processed tokens (est.)", f"{processed_tokens:,}")
                    c3.metric("Tokens saved", f"{saved_tokens:,}", f"{pct_tokens:.1f}%")

                    # === Save processed file to the app folder ===
                    base, _ext = os.path.splitext(up.name)
                    disk_filename = f"{base}_trimmed.pdf"
                    disk_path = unique_path_in_dir(APP_DIR, disk_filename)
                    try:
                        with open(disk_path, "wb") as f:
                            f.write(out_bytes)
                        st.info(f"Saved to: `{disk_path}`")
                    except Exception as e:
                        st.warning(f"Could not save to disk: {e}")

                    # Output preview (first up to max_preview_pages pages)
                    try:
                        out_doc = fitz.open(stream=out_bytes, filetype="pdf")
                        st.subheader("Output preview")
                        st.caption("A quick look at the processed file before downloading.")
                        prev_pages = min(max_preview_pages, out_doc.page_count)
                        cols = st.columns(min(3, prev_pages))
                        for i in range(prev_pages):
                            try:
                                img = render_page_image(out_doc, i, dpi=preview_dpi)
                                with cols[i % len(cols)]:
                                    st.image(img, caption=f"Out page {i+1}", use_container_width=True)
                            except Exception as re:
                                st.warning(f"Output preview failed for page {i+1}: {re}")
                        out_doc.close()
                    except Exception as e:
                        st.warning(f"Could not open output PDF for preview: {e}")

                    # Download (still available)
                    out_name_for_download = os.path.basename(disk_path)
                    st.download_button(
                        label="⬇️ Download processed PDF",
                        data=out_bytes,
                        file_name=out_name_for_download,
                        mime="application/pdf",
                        key=f"dl_{file_key}",
                        help="Download the processed PDF to your computer."
                    )

        # Close doc handle
        try:
            doc.close()
        except Exception:
            pass

st.divider()
st.caption(
    "Tip: For maximum token savings, prefer **KEEP** mode with only the pages you truly need. "
    "This app uses lossless optimizations (garbage collection, deflate, linearization fallback). "
    "Token estimates are based on extracted text; vision/image token costs are not included."
)
