import matplotlib
matplotlib.use("Agg")   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt

# Week 6 결과 그래프 3장 → docs/weekly/images/
# 숫자는 docs/weekly/week6.md의 결과표에서 가져왔다 (각 그래프 위에 출처 섹션을 적어 둠).
# 그래프에는 숫자만 들어가고 얼굴 사진은 없으므로 GitHub에 올려도 된다.

OUT = "docs/weekly/images"

# ---------- 공통 스타일 ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # 제목, 값
INK_2 = "#52514e"        # 축 이름, 보조 글씨
GRID = "#e4e3df"
BLUE = "#2a78d6"         # 범주색 1번
ORANGE = "#eb6834"       # 범주색 2번
MUTED = "#a3a29b"        # 비교 기준(예전 방식)

plt.rcParams.update({
    "font.family": "AppleGothic",     # 한글 표시 (macOS 기본 글꼴)
    "axes.unicode_minus": False,      # 한글 글꼴에서 마이너스 기호가 깨지지 않게
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def ref_line(ax, y, text, color=INK_2, style="--"):
    # 기준선: 얇은 점선 + 오른쪽 끝에 이름
    ax.axhline(y, color=color, linewidth=1, linestyle=style, zorder=1)
    ax.text(1.01, y, text, transform=ax.get_yaxis_transform(), color=color, va="center", fontsize=9)


# ---------- 1. 이동만 vs 회전·크기 EOT (② 결과, 사진 4장 평균, 한 번 실행) ----------
# 출처: week6.md ② 결과 (ArcFace), ② FaceNet 채점
data_1 = {
    "ArcFace (노이즈를 만든 모델)": {"none": 0.723, "success": 0.2,
                                 "이동만": [-0.043, -0.213], "회전·크기": [-0.357, -0.430]},
    "FaceNet (공격에 쓰지 않은 모델)": {"none": 0.629, "success": 0.365,
                                   "이동만": [0.284, 0.299], "회전·크기": [0.180, 0.035]},
}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
x = [0, 1]
w = 0.34
for ax, (title, d) in zip(axes, data_1.items()):
    for offset, method, color in [(-w / 2 - 0.01, "이동만", MUTED), (w / 2 + 0.01, "회전·크기", BLUE)]:
        bars = ax.bar([i + offset for i in x], d[method], width=w, color=color, label=method, zorder=2)
        for bar, v in zip(bars, d[method]):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.03 if v < 0 else v + 0.02, f"{v:.2f}",
                    ha="center", va="top" if v < 0 else "bottom", color=INK, fontsize=9,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ref_line(ax, d["none"], f"보호 안 함 {d['none']:.2f}")
    ref_line(ax, d["success"], f"성공선 {d['success']}", style=":")
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    ax.set_xticks(x, ["eps 8/255", "eps 12/255"])
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylim(-0.6, 0.85)
axes[0].set_ylabel("SimSwap 결과 vs 내 사진 (코사인 유사도)\n낮을수록 보호가 잘 됨")
axes[1].legend(loc="lower right", frameon=False, title="노이즈 만드는 방법", title_fontsize=9)
fig.suptitle("회전·크기 EOT가 이동만 쓴 방식보다 두 채점자 모두에서 더 낮다 (재정렬 후)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/week6_1_eot_compare.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------- 2. eps에 따른 효과와 화질 (③, ③-2, ④) ----------
# 출처: ④ seed 3개 평균(eps 0, 4, 5), ③·③-2·② 한 번 실행(eps 6, 8, 12). PSNR은 ③, ③-2, ④
eps = [0, 4, 5, 6, 8, 12]
arcface = [0.723, 0.012, -0.129, -0.219, -0.357, -0.430]
facenet = [0.629, 0.384, 0.297, 0.254, 0.180, 0.035]
averaged = [True, True, True, False, False, False]          # seed 3개 평균인가
psnr_eps = [4, 5, 6, 8, 12]
psnr = [38.6, 36.8, 35.6, 33.3, 30.3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
for ax in (ax1, ax2):
    ax.axvspan(5.5, 12.6, color="#f0efec", zorder=0)   # 눈에 보이는 구간
    ax.axvline(5, color=INK_2, linewidth=1, linestyle="--", zorder=1)
ax1.text(9.1, 0.78, "눈에 보임 (6: 약간, 8: 아지랑이)", color=INK_2, ha="center", fontsize=9)
ax1.text(4.9, 0.78, "선택: 5/255", color=INK, ha="right", fontsize=9)

for values, color, name, success in [(arcface, ORANGE, "ArcFace", 0.2), (facenet, BLUE, "FaceNet", 0.365)]:
    ax1.plot(eps, values, color=color, linewidth=2, zorder=3, label=name)
    for e, v, avg in zip(eps, values, averaged):
        ax1.plot(e, v, "o", markersize=8, zorder=4, color=color,
                 markerfacecolor=color if avg else SURFACE, markeredgewidth=2)
    ax1.axhline(success, color=color, linewidth=1, linestyle=":", zorder=1)
    ax1.text(1.01, success, f"{name} 성공선 {success}", transform=ax1.get_yaxis_transform(),
             color=color, va="center", fontsize=9)
ax1.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
ax1.plot([], [], "o", color=INK_2, markerfacecolor=INK_2, label="seed 3개 평균")
ax1.plot([], [], "o", color=INK_2, markerfacecolor=SURFACE, markeredgewidth=2, label="한 번 실행")
ax1.legend(loc="lower left", frameon=False, ncol=2)
ax1.set_ylabel("SimSwap 결과 vs 내 사진\n(코사인 유사도, 낮을수록 보호)")
ax1.set_ylim(-0.55, 0.85)
ax1.set_xlim(-0.6, 12.6)
ax1.set_title("eps를 키우면 보호는 강해지지만, 6/255부터 노이즈가 눈에 보인다",
              loc="left", color=INK, fontsize=12)

ax2.plot(psnr_eps, psnr, color=INK_2, linewidth=2, marker="o", markersize=8, zorder=3)
for e, p in zip(psnr_eps, psnr):
    ax2.text(e, p + 0.6, f"{p}", ha="center", color=INK, fontsize=9)
ax2.set_ylabel("화질 PSNR (dB)\n높을수록 원본과 비슷")
ax2.set_ylim(28, 41)
ax2.set_xlabel("노이즈 크기 eps (/255)")
ax2.set_xticks(eps)
fig.tight_layout()
fig.savefig(f"{OUT}/week6_2_eps_tradeoff.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------- 3. 변형별 효과 유지율 (⑤, eps 5/255, seed 3개 평균) ----------
# 출처: week6.md ⑤ 결과표
transforms = ["JPEG 90", "JPEG 75", "JPEG 50", "Resize 75% (bilinear)", "Resize 50% (bilinear)",
              "Resize 75% (area)", "Resize 50% (area)", "Crop 10%", "Blur 0.5", "Blur 1.0"]
keep_arc = [98, 95, 93, 97, 95, 98, 97, 95, 98, 96]
keep_fn = [93, 96, 89, 93, 95, 97, 99, 89, 99, 93]

fig, ax = plt.subplots(figsize=(9, 6))
y = list(range(len(transforms)))[::-1]
h = 0.36
ax.barh([i + h / 2 + 0.01 for i in y], keep_arc, height=h, color=ORANGE, label="ArcFace", zorder=2)
ax.barh([i - h / 2 - 0.01 for i in y], keep_fn, height=h, color=BLUE, label="FaceNet", zorder=2)
for i, a, f in zip(y, keep_arc, keep_fn):      # 가장 낮은 값만 숫자로 표시
    if f == min(keep_fn):
        ax.text(f + 1, i - h / 2, f"{f}%", va="center", color=INK, fontsize=9)
ax.axvline(50, color=INK_2, linewidth=1, linestyle=":", zorder=1)
ax.text(50, len(transforms) - 0.3, "유지 기준 50%", color=INK_2, ha="center", fontsize=9)
ax.set_yticks(y, transforms)
ax.set_xlim(0, 105)
ax.set_xlabel("효과 유지율 (%) = 변형 후 (보호 안 함 - 보호) / 변형 전 (보호 안 함 - 보호)")
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False)
ax.set_title("JPEG, 크기 조절, 자르기, 블러를 거쳐도 보호 효과가 89% 이상 남는다 (eps 5/255)",
             loc="left", color=INK, fontsize=12, pad=28)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(f"{OUT}/week6_3_robustness.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("저장 완료:", OUT)
