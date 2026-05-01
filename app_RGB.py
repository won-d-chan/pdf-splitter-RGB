import streamlit as st
import fitz
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import uniform_filter1d, label
import io
import zipfile
import os

st.title("📄 PDF 문제지 단 분리 도구")

with st.expander("📖 사용법 보기"):
    st.markdown("""
1. PDF 파일을 업로드하세요
2. **처리 시작** 버튼을 누르세요
3. 구분선 인식이 애매한 페이지는 미리보기와 선택지가 표시됩니다
   - 빨강: 세로선 기반 감지
   - 파랑: 밀도 기반 감지
   - 초록: 중앙값 (감지 실패 시)
   - 숫자가 클수록 구분선이 오른쪽으로 이동
4. **ZIP 다운로드** 버튼으로 결과물을 저장하세요

**출력 파일명:** `page01_left.png` / `page01_right.png`
    """)


# ── 기울기 보정 함수 ───────────────────────────
def detect_skew_by_projection(gray_array, angle_range=3.0, angle_step=0.2):
    if gray_array.dtype != np.uint8:
        gray_array = gray_array.astype(np.uint8)
    ret, binary = cv2.threshold(gray_array, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary is None or ret is None:
        return 0.0
    h, w = binary.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        score = rotated.sum(axis=1).astype(np.float64).var()
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def deskew(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    coarse = detect_skew_by_projection(gray, angle_range=5.0, angle_step=0.5)
    fine   = detect_skew_by_projection(gray,
                                       angle_range=abs(coarse) + 0.5,
                                       angle_step=0.05)
    angle = fine
    if abs(angle) < 0.05:
        return image_rgb, 0.0
    h, w = image_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    corrected = cv2.warpAffine(image_rgb, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected, angle


# ── 구분선 감지 A: 세로선 강화 기반 ─────────────
def find_divider_vertical(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    search_start = int(w * 0.40)
    search_end   = int(w * 0.60)

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    repair_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 80)))
    repaired       = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, repair_kernel)
    vert_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.35))))
    vertical_lines = cv2.morphologyEx(repaired, cv2.MORPH_OPEN, vert_kernel)

    col_score = vertical_lines.sum(axis=0) / 255.0
    col_score = uniform_filter1d(col_score, size=7)
    region    = col_score[search_start:search_end]

    if region.max() <= 0:
        return None, 0.0

    threshold      = max(region.max() * 0.45, h * 0.18)
    candidates_idx = np.where(region >= threshold)[0]

    if len(candidates_idx) == 0:
        return None, 0.0

    split_points = np.where(np.diff(candidates_idx) > 1)[0] + 1
    groups       = np.split(candidates_idx, split_points)

    best_x, best_score = None, -1.0

    for group in groups:
        if len(group) == 0:
            continue
        x          = int(np.mean(group)) + search_start
        line_score = float(np.max(col_score[group + search_start]))
        center_dist  = abs(x - w / 2)
        center_score = float(np.clip(1.0 - center_dist / (w * 0.20), 0, 1))
        gap, band    = 6, 35
        left_area    = binary[:, max(0, x-gap-band):max(0, x-gap)]
        right_area   = binary[:, min(w, x+gap):min(w, x+gap+band)]
        ws = 1.0 - ((left_area.mean() + right_area.mean()) / 2.0 / 255.0) \
             if left_area.size > 0 and right_area.size > 0 else 0.5
        final = 0.35 * np.clip(line_score / h, 0, 1) + 0.45 * center_score + 0.20 * ws
        if final > best_score:
            best_score, best_x = final, x

    if best_x is None:
        return None, 0.0
    if not (int(w * 0.40) <= best_x <= int(w * 0.60)):
        return None, best_score

    return best_x, best_score


# ── 구분선 감지 B: 텍스트 밀도 기반 ────────────
def find_divider_density(img_array):
    height, width = img_array.shape
    search_start  = int(width * 4 / 12)
    search_end    = int(width * 8 / 12)

    dark_ratio   = np.mean(img_array < 180, axis=0).astype(float)
    region       = dark_ratio[search_start:search_end]
    smoothed     = uniform_filter1d(region, size=20)
    relative_low = smoothed - region

    threshold = np.percentile(relative_low, 85)
    gap_mask  = (relative_low >= threshold).astype(int)
    labeled, num_features = label(gap_mask)

    center_x = len(region) // 2
    runs = []
    for i in range(1, num_features + 1):
        cols = np.where(labeled == i)[0]
        if len(cols) < 15:
            continue
        run_center = int(cols.mean())
        runs.append({"center": run_center,
                     "dist_from_mid": abs(run_center - center_x)})
    if not runs:
        return None

    best_run     = min(runs, key=lambda r: r["dist_from_mid"])
    rough_center = best_run["center"] + search_start
    fine_start   = max(0, rough_center - 50)
    fine_end     = min(width, rough_center + 50)
    return int(np.argmin(dark_ratio[fine_start:fine_end])) + fine_start


# ── 미리보기 이미지 생성 ────────────────────────
def make_preview(image_rgb, divider_a, divider_b, fallback):
    preview_arr = image_rgb.copy()
    if divider_a is not None:
        preview_arr[:, max(0, divider_a-2):divider_a+2] = [255, 0, 0]
    if divider_b is not None:
        preview_arr[:, max(0, divider_b-2):divider_b+2] = [0, 0, 255]
    if fallback is not None:
        preview_arr[:, max(0, fallback-2):fallback+2] = [0, 200, 0]
    preview_pil = Image.fromarray(preview_arr)
    ratio = 800 / preview_pil.height
    return preview_pil.resize((int(preview_pil.width * ratio), 800))


# ── 메인 UI ────────────────────────────────────
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

overlap_left  = st.sidebar.number_input("왼쪽 오버랩 (px)", value=5, min_value=0, max_value=100)
overlap_right = st.sidebar.number_input("오른쪽 오버랩 (px)", value=5, min_value=0, max_value=100)
tolerance     = st.sidebar.number_input("교차검증 허용 오차 (px)", value=15, min_value=0, max_value=100)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    st.write(f"총 **{len(doc)}** 페이지 감지됨")

    if st.button("▶ 처리 시작"):
        st.session_state["pages"]   = {}
        st.session_state["pending"] = []

        progress = st.progress(0)
        status   = st.empty()

        for page_num, page in enumerate(doc):
            status.write(f"처리 중... {page_num+1}/{len(doc)} 페이지")
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3)

            corrected_rgb, angle = deskew(image_rgb)
            img_gray = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2GRAY)
            width    = img_gray.shape[1]

            divider_a, confidence = find_divider_vertical(corrected_rgb)
            divider_b = find_divider_density(img_gray)

            # 교차검증
            if divider_a is not None and divider_b is not None:
                if abs(divider_a - divider_b) <= tolerance:
                    divider_x  = (divider_a + divider_b) // 2
                    status_str = "✅ 자동"
                else:
                    divider_x  = None
                    status_str = "⚠️ 불일치"
            elif divider_a is not None and confidence >= 0.55:
                divider_x  = divider_a
                status_str = "✅ 세로선만"
            elif divider_b is not None:
                divider_x  = divider_b
                status_str = "✅ 밀도만"
            else:
                divider_x  = None
                status_str = "⚠️ 둘 다 실패"

            st.session_state["pages"][page_num] = {
                "corrected_rgb": corrected_rgb,
                "img_gray":      img_gray,
                "divider_a":     divider_a,
                "divider_b":     divider_b,
                "divider_x":     divider_x,
                "confidence":    confidence,
                "angle":         angle,
                "status":        status_str,
            }

            if divider_x is None:
                st.session_state["pending"].append(page_num)

            progress.progress((page_num + 1) / len(doc))

        status.write("✅ 분석 완료!")
        st.session_state["analyzed"] = True

# ── 수동 확인 필요한 페이지 처리 ────────────────
if st.session_state.get("analyzed"):
    pending = st.session_state.get("pending", [])

    if pending:
        st.subheader("⚠️ 수동 확인 필요 페이지")
        for page_num in pending:
            data  = st.session_state["pages"][page_num]
            width = data["img_gray"].shape[1]

            st.write(f"**{page_num+1}페이지** — {data['status']}")
            st.write(
                f"빨강(세로선): `{data['divider_a']}px` / "
                f"파랑(밀도): `{data['divider_b']}px` / "
                f"초록(중앙): `{width//2}px`"
            )

            preview = make_preview(
                data["corrected_rgb"],
                data["divider_a"],
                data["divider_b"],
                width // 2
            )
            st.image(preview, use_container_width=True)

            choice = st.radio(
                f"{page_num+1}페이지 구분선 선택",
                options=["빨강(세로선)", "파랑(밀도)", "초록(중앙값)", "직접입력"],
                key=f"radio_{page_num}"
            )
            if choice == "빨강(세로선)" and data["divider_a"] is not None:
                st.session_state["pages"][page_num]["divider_x"] = data["divider_a"]
            elif choice == "파랑(밀도)" and data["divider_b"] is not None:
                st.session_state["pages"][page_num]["divider_x"] = data["divider_b"]
            elif choice == "초록(중앙값)":
                st.session_state["pages"][page_num]["divider_x"] = width // 2
            elif choice == "직접입력":
                val = st.number_input(
                    f"{page_num+1}페이지 직접입력 (px)",
                    min_value=1, max_value=width-1,
                    value=width//2,
                    key=f"manual_{page_num}"
                )
                st.session_state["pages"][page_num]["divider_x"] = val

    # ── 최종 저장 ──────────────────────────────
    st.subheader("📥 저장 및 다운로드")

    if st.button("💾 분리 이미지 생성 및 다운로드"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for page_num, data in st.session_state["pages"].items():
                divider_x    = data["divider_x"]
                corrected_rgb = data["corrected_rgb"]
                h, w_full    = corrected_rgb.shape[:2]

                if divider_x is None:
                    divider_x = w_full // 2

                left_end    = min(divider_x + overlap_left, w_full)
                right_start = max(divider_x - overlap_right, 0)

                left_img  = Image.fromarray(corrected_rgb[:, :left_end])
                right_img = Image.fromarray(corrected_rgb[:, right_start:])

                for side, out_img in [("left", left_img), ("right", right_img)]:
                    buf = io.BytesIO()
                    out_img.save(buf, format="PNG")
                    zf.writestr(f"page{page_num+1:02d}_{side}.png",
                                buf.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="📦 ZIP 다운로드",
            data=zip_buffer,
            file_name="분리된이미지.zip",
            mime="application/zip"
        )