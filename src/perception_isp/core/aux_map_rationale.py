"""Korean design rationale and implemented equations for public Aux maps.

This module complements :mod:`perception_isp.core.aux_map_catalog`.  The public
catalog describes the output contract; this catalog explains the failure mode,
implemented equation, and evidentiary boundary behind every map.  Keeping the
two name sets synchronized makes the long-form report fail loudly when a map is
added or removed without documenting its rationale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from perception_isp.core.aux_map_catalog import AUX_MAP_SPECS


@dataclass(frozen=True)
class AuxMapRationale:
    name: str
    problem_situation: str
    formula: str
    why_it_helps: str
    design_basis: str
    interpretation_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r(
    name: str,
    problem_situation: str,
    formula: str,
    why_it_helps: str,
    design_basis: str,
    interpretation_boundary: str,
) -> AuxMapRationale:
    return AuxMapRationale(
        name=name,
        problem_situation=problem_situation,
        formula=formula,
        why_it_helps=why_it_helps,
        design_basis=design_basis,
        interpretation_boundary=interpretation_boundary,
    )


AUX_MAP_RATIONALES: tuple[AuxMapRationale, ...] = (
    _r(
        "lens_gain",
        "렌즈 비네팅 때문에 주변부가 어둡고, 같은 밝기로 보정할수록 주변부 노이즈도 함께 증폭되는 상황.",
        "L = resize_nearest(L_cal, H, W);  x_corr = x_gain_corrected · L;  Var_out = Var_raw · L²",
        "신호 보정량 L과 그에 따른 분산 증폭 L²을 함께 노출하면, 모델이 중앙부와 강하게 보정된 주변부를 같은 신뢰도로 취급하지 않게 할 수 있다.",
        "측정된 lens-shading calibration의 직접 전달과 선형 gain의 분산 전파 법칙을 결합했다.",
        "L_cal이 없을 때 L=1은 중립 기본값일 뿐, 실제 렌즈의 shading이 없다는 측정 결과가 아니다. 신호 보정 후 다시 L²을 쓰는 분산 모델이므로 sensor profile 계수의 정의와 일치해야 한다.",
    ),
    _r(
        "defect_confidence",
        "hot/dead/stuck pixel을 이웃값으로 교체했지만 후단이 그 픽셀을 원래 센서 관측값으로 오인하는 상황.",
        "D(p)=0 if p∈defect_table else 1;  x_corr(p)=box_filter₃×₃(x)(p) for p∈defect_table",
        "교체된 샘플을 명시적으로 표시해 edge, health 또는 학습 입력이 보간값에 과도한 의미를 부여하지 않게 한다.",
        "센서 defect table과 국소 평균 보간을 사용하는 calibration-derived 신뢰도다.",
        "모두 1인 map은 defect가 없다는 증명이 아니라 유효한 defect table 항목이 없었다는 뜻일 수 있다. 3×3 mean에는 교체 전 중심값도 포함된다.",
    ),
    _r(
        "hdr_exposure_source",
        "장면의 밝은 부분과 어두운 부분에서 서로 다른 노출이 지배적으로 사용되는데, fusion 뒤에는 그 출처가 사라지는 상황.",
        "rᵢ=xᵢ/(tᵢ/maxⱼtⱼ);  wᵢ=max(qᵢ·nᵢ·lᵢ·mᵢ,0);  source=argmaxᵢ(wᵢ), Σwᵢ=0이면 argminᵢ(xᵢ)",
        "어느 plane이 픽셀에 가장 크게 기여했는지 감사할 수 있어 highlight/shadow 전환, exposure seam, 실패 위치를 분석할 수 있다.",
        "노출시간으로 정규화한 radiance proxy와 SNR·포화·저신호·plane 일치도 가중치를 결합한 프로젝트의 auditable fusion heuristic이다.",
        "가장 큰 weight의 index이며 hard selection 결과가 아니다. single exposure에서는 항상 0이다.",
    ),
    _r(
        "saturation",
        "하나 이상의 노출 plane이 포화되어 그 위치의 색·경계 정보가 신뢰하기 어려운 상황.",
        "S(p)=maxᵢ 1[xᵢ(p) ≥ τ_sat],  τ_sat=0.985 (default)",
        "포화 위치를 binary gate로 제공해 edge/color confidence가 포화 구조를 강한 증거로 오인하는 것을 줄인다.",
        "센서의 유효 white 근처에서 clipping risk를 표시하는 표준 thresholding 원리를 사용한다.",
        "S=1은 입력 plane 중 하나가 포화됐다는 뜻이지 fused HDR까지 복구 불가능하다는 뜻은 아니다. 짧은 plane이 복구해도 1이라 현재 edge/color gate는 보수적이다.",
    ),
    _r(
        "clipping_distance",
        "binary saturation 직전의 연속적인 highlight headroom 차이를 구분해야 하는 상황.",
        "C_clip=clip((τ_sat−maxᵢxᵢ)/max(τ_sat,ε), 0, 1)",
        "0에 가까울수록 clipping 위험이 높다는 연속 신호를 제공하므로 binary saturation보다 부드러운 gating이 가능하다.",
        "정규화된 sensor code의 threshold까지 남은 거리를 무차원화한 프로젝트 heuristic이다.",
        "가장 밝은 입력 plane 기준이며 fused radiance의 물리적 headroom이나 복구 가능성을 뜻하지 않는다. 한 plane만 포화돼도 0이다.",
    ),
    _r(
        "hdr_confidence",
        "노출들이 포화·저신호·noise·불일치 때문에 fusion에 거의 기여하지 못하는 위치를 찾아야 하는 상황.",
        "qᵢ=(rᵢ/√(a·max(rᵢ,0)+b+ε)); q̄ᵢ=qᵢ/(qᵢ+1);  H=clip(Σᵢwᵢ/E,0,1)",
        "실제로 사용 가능한 exposure support의 합을 요약해 output validity와 color confidence를 낮출 수 있다.",
        "Poisson–Gaussian형 SNR, non-saturation, low-signal penalty, inter-plane consistency를 곱한 confidence-weighted estimation이다.",
        "확률로 calibration된 confidence나 HDR 품질 ground truth가 아니며, a·b와 exposure metadata가 틀리면 의미도 함께 틀어진다.",
    ),
    _r(
        "ghost_motion_artifact",
        "여러 노출 사이의 물체/ego motion, flicker 또는 잘못된 exposure 정규화가 한 픽셀에서 서로 다른 radiance를 만드는 상황.",
        "G=clip(stdᵢ(rᵢ)/(meanᵢ|rᵢ|+ε),0,1);  E=1이면 G=0",
        "plane disagreement를 직접 보이면 fusion edge를 낮추거나 temporal/HDR stress 실패를 위치별로 드러낼 수 있다.",
        "coefficient-of-variation 형태의 scale-normalized dispersion을 사용한 진단 heuristic이다.",
        "registration·optical flow·occlusion 처리가 없으므로 motion, JPEG 차이, AE 오차, noise를 분리하지 못하며 deghosting 결과도 아니다.",
    ),
    _r(
        "luma",
        "display tone mapping 전의 구조 밝기를 빠르게 사용해야 하며 RGB 재구성 artifact에 덜 의존하고 싶은 상황.",
        "Y_proxy=G_interp (Bayer); C_interp (RCC*); mean(RGB_corrected) (RGB-IR); x_fused (MONO)",
        "센서에 가까운 dense intensity를 edge, temporal state, fast path에 제공해 display rendering과 구조 분석을 분리한다.",
        "CFA별로 가장 직접적인 intensity channel을 선택하는 decoder-specific proxy 설계다.",
        "Bayer의 green은 colorimetric luminance가 아니며 CFA mode가 달라지면 값의 생성 방식도 달라진다.",
    ),
    _r(
        "clear_channel",
        "RCCB/RCCC/RCCG 센서의 clear pixel이 저조도 구조를 담지만 RGB로 변환하면서 그 직접 관측이 사라지는 상황.",
        "C=weighted_interpolate(x_fused·M_C,M_C); MONO이면 C=x_fused; Bayer/RGB-IR이면 C=0",
        "clear sample의 높은 광자 효율 정보를 별도 채널로 보존해 저조도 구조와 색 불확실성을 후단이 구분할 수 있다.",
        "CFA mask에 따른 측정 sample 보존과 normalized weighted interpolation을 사용한다.",
        "native clear/mono CFA에서만 물리적이다. Bayer CameraE2E 또는 RGB bridge가 0을 낸다고 clear 정보가 필요 없다는 뜻은 아니다.",
    ),
    _r(
        "ir_channel",
        "RGB-IR 센서에서 IR 조명과 visible color leakage를 후단이 구분해야 하지만 RGB 변환 후 IR 관측이 사라지는 상황.",
        "IR=weighted_interpolate(x_fused·M_IR,M_IR); non-RGB-IR CFA이면 IR=0",
        "IR 측정값을 독립적으로 전달해 야간 IR illumination, crosstalk, visible-color 신뢰도를 조건화할 수 있다.",
        "RGB-IR CFA mask 기반 interpolation과 별도 4→3 crosstalk calibration 경로를 사용한다.",
        "native RGB-IR와 유효한 crosstalk calibration에서만 물리적이며 RGB-derived pseudo-RAW는 IR 관측을 만들 수 없다.",
    ),
    _r(
        "demosaic_confidence",
        "색이 측정되지 않은 위치를 보간할 때 방향 선택이 모호하거나 고주파 영역에서 zipper/false-color가 생기는 상황.",
        "d₀=clip(0.42+0.28·support_norm+0.22·|g_h−g_v|/(g_h+g_v+ε)+0.08·(1−contrast_p98),0,1); d=clip(d₀·E_conf+(1−E_strength)·0.5,0,1)",
        "CFA support·방향 분리·contrast·후단 edge reliability를 함께 주어 artifact suppression이 flat 영역과 신뢰 경계를 다르게 처리하게 한다.",
        "edge-aware Bayer interpolation의 선택 모호도와 local support를 사용한 decoder heuristic이다. bilinear/RCC*/RGB-IR/MONO는 각 decoder의 d₀를 쓴다.",
        "demosaic ground truth로 calibration된 확률이 아니다. Bayer mask 합은 거의 항상 1이라 support 항의 공간 변별력이 작고, later edge confidence가 다시 섞인다.",
    ),
    _r(
        "color_confidence",
        "저 SNR, 포화, 약한 HDR support 또는 심한 channel imbalance에서 색 feature가 object cue보다 artifact가 되는 상황.",
        "μ_c=mean(RGB_c); W=clip(1−(max μ_c−min μ_c)/(mean μ_c+ε),0,1); C_color=clip(SNR_map·(1−S)·W·H,0,1)",
        "색을 신뢰하기 위한 네 조건을 곱해 하나라도 나쁘면 chroma 의존도를 줄일 수 있게 한다.",
        "signal reliability와 clipping gate에 global white-balance plausibility와 HDR support를 결합한 project heuristic이다.",
        "강한 단색 장면도 global imbalance로 불리해질 수 있고, color matrix/WB calibration이 틀리면 confidence도 편향된다.",
    ),
    _r(
        "log_r_over_g",
        "전체 밝기 변화와 exposure scale에 덜 민감한 red chromaticity cue가 필요한 상황.",
        "L_RG=log(max(R,ε)/max(G,ε)),  [R,G,B]=clip(M_camera·RGB_decode,0,4)",
        "공통 multiplicative illumination scale이 ratio에서 상쇄되어 색 변화와 밝기 변화를 어느 정도 분리한다.",
        "로그 색도(log-chromaticity)의 scale invariance를 사용한다.",
        "작은 G, clipping, demosaic error, illuminant spectrum과 부정확한 color matrix에 민감하며 illumination invariant를 보장하지 않는다.",
    ),
    _r(
        "log_b_over_g",
        "전체 밝기 변화와 exposure scale에 덜 민감한 blue chromaticity cue가 필요한 상황.",
        "L_BG=log(max(B,ε)/max(G,ε)),  [R,G,B]=clip(M_camera·RGB_decode,0,4)",
        "공통 multiplicative illumination scale이 ratio에서 상쇄되어 색 변화와 밝기 변화를 어느 정도 분리한다.",
        "로그 색도(log-chromaticity)의 scale invariance를 사용한다.",
        "작은 G, clipping, demosaic error, illuminant spectrum과 부정확한 color matrix에 민감하며 illumination invariant를 보장하지 않는다.",
    ),
    _r(
        "ir_contamination",
        "IR 에너지가 visible RGB를 왜곡하는 위치에서 색 feature를 그대로 믿는 상황.",
        "C_IR=clip(IR/(mean(R,G,B)+IR+ε),0,1)",
        "총 visible+IR proxy 중 IR 비율을 제공해 visible color feature를 위치별로 낮출 수 있다.",
        "두 nonnegative 성분의 bounded fraction을 쓰는 spectral contamination heuristic이다.",
        "genuine RGB-IR 입력에서만 의미가 있다. Bayer에서 0은 실제 IR-cut 성능을 측정한 결과가 아니다.",
    ),
    _r(
        "noise_variance",
        "어두운 영역, 고온·장노출, lens-shading 보정 주변부가 같은 pixel variance를 갖는다고 가정하면 edge와 denoise가 실패하는 상황.",
        "V_raw=a·max(x,0)+b+d·max(T−25,0)·t+q+V_cal;  V=max(V_raw·L²,ε)",
        "signal-dependent shot noise와 additive source, dark current, calibration residual, lens gain 증폭을 공간별 분산으로 전달한다.",
        "RAW 센서의 Poisson–Gaussian noise model에 temperature/exposure dark-current와 calibration residual을 추가한 software reference다.",
        "계수가 실측값이 아니면 상대적 proxy다. HDR weight별 분산을 전파하지 않고 fused signal에 재적용하며 dark-current는 첫 exposure time만 사용한다. CameraE2E 기본값은 nuScenes calibration이 아니다.",
    ),
    _r(
        "snr_map",
        "절대 signal 또는 variance만으로는 밝지만 noisy한 영역과 어둡지만 안정된 영역의 reliability를 비교하기 어려운 상황.",
        "SNR=x/√(V+ε);  SNR_map=clip(SNR/(SNR+1),0,1)",
        "signal-to-noise ratio를 bounded channel로 만들어 edge/color gate와 compact RGB+Aux input이 안정적으로 사용할 수 있게 한다.",
        "추정 분산으로 표준화한 classical SNR과 단조 bounded transform을 사용한다.",
        "실측 SNR이 아니며 noise calibration이 부정확하면 값도 낙관적 또는 비관적으로 편향된다.",
    ),
    _r(
        "noise_normalized_gradient",
        "작은 gradient가 실제 구조인지 예상 noise fluctuation인지 구분해야 하는 상황.",
        "g=√((∂Y/∂x)²+(∂Y/∂y)²);  G_N=g/√(2V+ε)",
        "두 축 차분의 noise variance를 2V로 근사해 gradient를 noise scale로 표준화하므로 구조 대비 noise의 상대 크기를 제공한다.",
        "variance-normalized residual/gradient라는 통계적 검정 아이디어를 적용했다.",
        "인접 noise의 독립·등분산 근사이며 texture, aliasing, demosaic artifact도 높은 값을 만들 수 있다.",
    ),
    _r(
        "edge_strength",
        "luma만의 gradient는 색 경계를 놓치고, raw gradient는 noise를 강한 경계로 오인하는 상황.",
        "J=Σ_{c∈{Y,R−G,B−G}} α_c/V · [g_x²  g_xg_y; g_xg_y  g_y²], α={SNR,0.5SNR,0.5SNR}; E=√max(λ₁(J),0)/max_frame",
        "noise/SNR로 가중한 다채널 structure tensor가 luminance와 chroma 구조를 모으고, 최대 eigenvalue가 지배적 방향의 경계 세기를 나타낸다.",
        "classical second-moment/structure-tensor eigen-analysis를 sensor reliability weighting으로 확장한 설계다.",
        "별도 neighborhood smoothing 없이 pixel-local gradient outer product를 사용한다. frame maximum 정규화라 run 간 절대 비교가 불가능하고 극단값과 calibration 오차에 민감하다.",
    ),
    _r(
        "edge_orientation",
        "경계에 수직인 dominant gradient 방향에 따라 filter 또는 geometric feature를 조건화해야 하는 상황.",
        "θ=0.5·atan2(2J_xy, J_xx−J_yy+ε)",
        "pixel-local second-moment tensor의 주축을 π-periodic half-angle로 표현해 dominant gradient/edge-normal 방향을 보존한다.",
        "2×2 symmetric structure tensor의 principal-axis 해석을 그대로 사용한다.",
        "edge tangent가 아니라 gradient/edge-normal 방향이다. flat/noisy/isotropic texture에서는 λ₁≈λ₂라 불안정하므로 strength/confidence와 함께 해석해야 한다.",
    ),
    _r(
        "edge_confidence",
        "강한 gradient라도 noise, saturation, optical blur 또는 isotropic texture에서 생겼다면 object boundary로 신뢰하기 어려운 상황.",
        "A=(λ₁−λ₂)/(λ₁+λ₂+ε);  E_conf=clip(A·MTF·exp(−0.5σ_PSF²)·(1−S)·SNR_map,0,1)",
        "방향성·광학 전달·포화·신호 신뢰도 중 하나라도 부족하면 confidence가 낮아져 artifact를 억제하고 supported edge를 남긴다.",
        "structure anisotropy와 calibrated optics priors를 multiplicative confidence gating으로 결합했다.",
        "object-boundary 확률이 아니고 MTF/PSF/noise calibration이 없으면 일부 항은 neutral default다.",
    ),
    _r(
        "edge_evidence",
        "strength만 쓰면 noise edge가, confidence만 쓰면 실제 gradient가 약한 위치도 통과할 수 있는 상황.",
        "U(z)=clip((z−p₁(z))/(p₉₉(z)−p₁(z)),0,1);  E_evidence=√(U(E_strength)·U(E_conf))",
        "세기와 신뢰도가 동시에 높을 때만 큰 geometric mean을 주어 compact DNN/proposal feature로 사용하기 쉽게 한다.",
        "robust percentile normalization과 AND-like geometric mean을 사용한 project fusion heuristic이다.",
        "p1/p99가 frame마다 달라 값은 run-relative이며 probability calibration이 아니다.",
    ),
    _r(
        "edge_type",
        "밝기 경계와 색만 바뀌는 경계가 다른 artifact/semantic 의미를 갖는데 하나의 magnitude로 합쳐지는 상황.",
        "L=|∇Y|/max_frame; C=max(|∇(R−G)|,|∇(B−G)|)/max_frame; type={0,1,2,3} for (L>0.12,C>0.12)",
        "luma-only, chroma-only, both를 분리해 color aliasing에 민감한 위치와 구조 경계를 구분할 수 있다.",
        "normalized gradient의 2-bit categorical diagnostic으로 설계했다.",
        "0.12는 고정된 project threshold이고 frame-relative normalization이라 보편적인 edge taxonomy가 아니다.",
    ),
    _r(
        "blur_focus_confidence",
        "detail이 필요한 위치에서 defocus/motion blur/저대비로 강하고 신뢰할 경계가 부족한 상황.",
        "T=p_q(E_strength), q=90 default;  F=0 if T≤ε else clip(0.5·E_strength/T+0.5·E_conf,0,1)",
        "frame의 강한-edge percentile에 대한 상대 strength와 edge reliability를 평균해 focus-like support를 국소적으로 요약한다.",
        "sharp-edge energy를 focus proxy로 쓰는 classical heuristic에 sensor-aware edge confidence를 결합했다.",
        "focus 측정기가 아니며 texture 부족, motion blur, defocus, haze, low contrast를 분리하지 못한다.",
    ),
    _r(
        "mtf_confidence",
        "렌즈/센서 위치별 해상력 저하 때문에 같은 gradient도 전달 가능한 spatial detail이 다른 상황.",
        "MTF=resize_nearest(MTF_cal,H,W);  MTF_cal is None이면 MTF=1",
        "측정된 spatial transfer prior를 edge confidence에 직접 곱해 알려진 저해상력 영역의 신뢰를 낮춘다.",
        "measured MTF calibration의 passthrough다. 알고리즘이 이미지에서 MTF를 추정하지 않는다.",
        "uniform 1은 완벽한 MTF 측정 결과가 아니라 calibration 부재 시 neutral default다. 현재 입력 map은 별도 finite/[0,1] 검증 없이 전달된다.",
    ),
    _r(
        "psf_sigma",
        "공간적으로 달라지는 optical blur width를 후단이 모르면 같은 edge를 동일하게 취급하는 상황.",
        "σ_PSF=max(resize_nearest(σ_cal,H,W),0);  σ_cal is None이면 σ_PSF=0",
        "calibration이 정의한 local blur scale을 그대로 노출해 optics-aware gating과 분석이 가능해진다.",
        "measured/fitted PSF width calibration의 passthrough다.",
        "단위는 calibration-defined sensor pixels이며 0 default는 diffraction-free optics를 측정했다는 뜻이 아니다. 현재 finite 검증은 별도로 하지 않는다.",
    ),
    _r(
        "psf_blur_confidence",
        "PSF width는 양의 비한정 값이라 다른 bounded confidence와 직접 결합하기 어려운 상황.",
        "B_PSF=clip(exp(−0.5·σ_PSF²),0,1)",
        "Gaussian PSF scale이 커질수록 단조롭게 edge 신뢰를 낮추는 bounded prior로 변환한다.",
        "Gaussian blur의 frequency attenuation 형태를 단순화한 calibration-derived mapping이다.",
        "실제 MTF 전체 곡선을 대신하지 않으며 σ calibration이 없을 때 1은 neutral default다.",
    ),
    _r(
        "psf_edge_likelihood",
        "구조적으로 신뢰되는 edge라도 multi-exposure disagreement가 크면 HDR ghost일 가능성이 있는 상황.",
        "P_edge=clip(E_conf−G_ghost,0,1)",
        "reliable edge에서 inter-plane disagreement를 빼 compact edge/optics cue가 ghost-prone 경계를 낮추게 한다.",
        "edge reliability와 HDR inconsistency를 결합한 project diagnostic heuristic이다.",
        "역사적 이름과 달리 PSF 확률분포로 fitting한 likelihood가 아니며 single exposure에서는 ghost penalty가 0이다.",
    ),
    _r(
        "rolling_row_fraction",
        "rolling shutter에서 행마다 촬영 시점이 달라 motion/flicker artifact 위치가 row에 의존하는 상황.",
        "R=linspace(0,1,H); bottom_to_top이면 R←reverse(R)",
        "모델과 진단기가 normalized readout phase를 받아 row-dependent artifact를 조건화할 수 있다.",
        "sensor readout metadata에서 직접 만든 dimensionless timing coordinate다.",
        "row timestamp는 Trace에 별도 제공되지만 이 map 자체는 motion warp나 rolling-shutter 보정을 수행하지 않는다.",
    ),
    _r(
        "temporal_difference",
        "같은 camera chain에서 밝기/물체/ego motion/flicker가 변했는데 single-frame feature만으로 불안정을 보지 못하는 상황.",
        "Y_t=mean(RGB_perception,t);  Δ_t=|Y_t−Y_{t−1}|; compatible previous state가 없으면 Δ_t=0",
        "가장 단순한 photometric residual을 fast path에 제공해 변화·motion·frozen-frame 이상을 표시한다.",
        "PreviousFrameState를 이용한 first-order temporal residual이다.",
        "registration이 없어 ego/object motion, AE, JPEG artifact와 flicker가 섞인다. PreviousFrameState 자체에는 camera_id 검증이 없고, state 부재/shape mismatch의 0은 관측된 정지가 아닌 neutral fallback이다.",
    ),
    _r(
        "temporal_consistency",
        "raw difference는 비한정 범위라 안정성 gate로 직접 쓰기 어려운 상황.",
        "T_cons=exp(−Δ_t/max(τ_time,ε)),  τ_time=0.15 default; previous state가 없으면 1",
        "작은 변화에는 높은 값, 큰 변화에는 빠르게 낮은 값을 주는 bounded similarity로 temporal gating을 단순화한다.",
        "Laplace-like exponential photometric kernel을 사용한 project similarity heuristic이다.",
        "correspondence나 optical-flow confidence가 아니며 threshold는 sensor/scene별 calibration이 필요하다. state 부재 때 1은 관측된 안정성이 아닌 neutral fallback이다.",
    ),
    _r(
        "led_flicker_confidence",
        "밝은 LED가 시간적으로 변하거나 multi-exposure plane에서 불일치하지만 일반 motion residual과 구분할 단서가 필요한 상황.",
        "B=clip((Y−0.65)/0.35,0,1); H_flicker=clip(G_ghost·S,0,1); F=clip((1−T_cons)·B+H_flicker,0,1)",
        "밝은 영역의 temporal instability와 포화된 HDR disagreement를 모아 runtime control이 flicker-like risk를 볼 수 있게 한다.",
        "brightness gating, temporal residual, HDR disagreement를 결합한 targeted project heuristic이다.",
        "LED classifier가 아니며 motion, AE 변화, compression, reflection도 높은 값을 만들 수 있다.",
    ),
    _r(
        "light_source_confidence",
        "밝은 픽셀 중 안정적인 light-source-like 영역과 flicker/highlight artifact를 어느 정도 분리하고 싶은 상황.",
        "B=clip((Y−0.65)/0.35,0,1);  L_src=clip(B·(1−0.5F),0,1)",
        "brightness prior에 flicker penalty를 적용해 안정적인 밝은 영역을 유지하면서 불안정 highlight의 신뢰를 낮춘다.",
        "thresholded brightness와 temporal penalty를 조합한 diagnostic heuristic이다.",
        "lamp, reflection, sky, saturated surface를 semantic하게 구분하지 못하며 configured linear scale에 의존한다.",
    ),
)


AUX_MAP_RATIONALE_BY_NAME: dict[str, AuxMapRationale] = {
    row.name: row for row in AUX_MAP_RATIONALES
}


def validate_aux_map_rationales(names: Iterable[str] | None = None) -> None:
    """Fail when the rationale set and public Aux contract drift apart."""

    rationale_names = [row.name for row in AUX_MAP_RATIONALES]
    if len(rationale_names) != len(set(rationale_names)):
        raise ValueError("Aux-map rationale catalog contains duplicate names")
    expected = {row.name for row in AUX_MAP_SPECS}
    actual = set(rationale_names)
    if names is not None:
        expected = {str(name) for name in names}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "Aux-map rationale catalog drift: "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )


def aux_map_rationale_json() -> list[dict[str, Any]]:
    validate_aux_map_rationales()
    return [row.to_dict() for row in AUX_MAP_RATIONALES]


validate_aux_map_rationales()


__all__ = [
    "AUX_MAP_RATIONALES",
    "AUX_MAP_RATIONALE_BY_NAME",
    "AuxMapRationale",
    "aux_map_rationale_json",
    "validate_aux_map_rationales",
]
