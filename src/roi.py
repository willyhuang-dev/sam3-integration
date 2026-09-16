"""ROI 幾何 → token 索引 → 逐層遮罩與 RoPE 索引。

刻意只依賴 torch 與 math:本模組是全部索引算術的所在地,也是最容易寫錯的地方
(row%24 這類 off-by-one)。與 transformers 解耦才能單獨測到底。

策略 B 用的 bucket_table 不在此實作 —— 它的消費者在階段 3,現在寫就是沒有
測試保護的推測程式碼。見 docs/design.md 3.1。
"""
import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Grid:
    """token 網格的幾何。size/window 以 token 計,patch 以像素計。"""

    size: int = 72
    window: int = 24
    patch: int = 14

    @property
    def n_tokens(self) -> int:
        return self.size * self.size

    @property
    def wins_per_side(self) -> int:
        return self.size // self.window

    @property
    def n_windows(self) -> int:
        return self.wins_per_side ** 2

    @property
    def canvas(self) -> int:
        """畫布邊長(像素)。SAM3 是 72 × 14 = 1008。"""
        return self.size * self.patch


SAM3_GRID = Grid()
TOY_GRID = Grid(size=8, window=4, patch=14)   # cost_model.md 的推導用網格


def grid_rc(idx: Tensor, grid: Grid = SAM3_GRID) -> tuple[Tensor, Tensor]:
    """token 索引 → (row, col)。"""
    return torch.div(idx, grid.size, rounding_mode="floor"), idx % grid.size


def window_id(idx: Tensor, grid: Grid = SAM3_GRID) -> Tensor:
    """token 索引 → 它所屬 window 的編號(row-major)。"""
    row, col = grid_rc(idx, grid)
    wr = torch.div(row, grid.window, rounding_mode="floor")
    wc = torch.div(col, grid.window, rounding_mode="floor")
    return wr * grid.wins_per_side + wc


def rope_indices(idx: Tensor, grid: Grid = SAM3_GRID) -> tuple[Tensor, Tensor]:
    """回傳 (i_global, i_window),用來 index_select 各層現成的 RoPE buffer。

    global 層的 buffer 是 size×size(scale = window/size),索引就是 token index。
    windowed 層的 buffer 是 window×window(scale = 1.0),索引是 window 內座標。
    因此三角函數與 rotary_scale 一行都不用改。見 cost_model.md 7。
    """
    row, col = grid_rc(idx, grid)
    i_window = (row % grid.window) * grid.window + (col % grid.window)
    return idx, i_window


def boxes_to_token_idx(
    boxes: Tensor,
    orig_size: tuple[int, int],
    dilate: int = 0,
    grid: Grid = SAM3_GRID,
) -> Tensor:
    """原圖座標的框(可多個)→ 涵蓋到的 token 索引聯集。

    Args:
        boxes: (M, 4) xyxy,**原圖像素座標**。
        orig_size: (H, W) 原圖尺寸。
        dilate: 四邊各向外膨脹幾個 token。
        grid: token 網格幾何。

    Returns:
        排序去重的 int64 索引。空輸入回傳 shape (0,) 的 int64 張量。

    內含原圖 → canvas 的**非等比**縮放(SAM3 的前處理直接 resize 成正方形,
    x 與 y 的縮放比不同)。
    """
    height, width = orig_size
    canvas = grid.canvas
    pieces = []
    for x0, y0, x1, y1 in boxes.reshape(-1, 4).tolist():
        # 原圖像素 → canvas 像素(逐軸獨立縮放)
        cx0, cx1 = x0 * canvas / width, x1 * canvas / width
        cy0, cy1 = y0 * canvas / height, y1 * canvas / height
        # canvas 像素 → token 索引範圍(含端點)
        c0 = int(math.floor(cx0 / grid.patch)) - dilate
        c1 = int(math.ceil(cx1 / grid.patch)) - 1 + dilate
        r0 = int(math.floor(cy0 / grid.patch)) - dilate
        r1 = int(math.ceil(cy1 / grid.patch)) - 1 + dilate
        c0, c1 = max(0, c0), min(grid.size - 1, c1)
        r0, r1 = max(0, r0), min(grid.size - 1, r1)
        if c1 < c0 or r1 < r0:
            continue
        rows = torch.arange(r0, r1 + 1, dtype=torch.long)
        cols = torch.arange(c0, c1 + 1, dtype=torch.long)
        pieces.append((rows[:, None] * grid.size + cols[None, :]).reshape(-1))
    if not pieces:
        return torch.zeros(0, dtype=torch.long)
    return torch.unique(torch.cat(pieces))


def pack(
    idx_groups: list[Tensor],
    n_budget: int,
    grid: Grid = SAM3_GRID,
) -> tuple[Tensor, int, int]:
    """把依優先序排列的索引群組打包成固定長度 n_budget 的張量。

    Args:
        idx_groups: 索引群組 list,**高優先在前**。產線路徑是 [右手, 左手, 人];
            評測路徑是 GT 框面積小者優先(小物件更需要那些 token)。
        n_budget: 靜態 token 預算 N。TensorRT 要求形狀固定,所以這是編譯期常數。
        grid: token 網格幾何。

    Returns:
        (roi_token_idx, n_valid, n_dropped)
        roi_token_idx: (n_budget,) int64。**真索引緊密排在前面**,尾端補 0。
            這個「緊密靠前」的性質是 n_valid 純量介面成立的前提。
        n_valid: 有效 token 數,保證 >= 1。
        n_dropped: 因預算不足而丟棄的索引數。

    組內按索引升序、截斷取前段 —— 任意但確定性。丟棄量由 n_dropped 回報,
    呼叫端應記錄到結果 JSON。
    """
    seen = torch.zeros(grid.n_tokens, dtype=torch.bool)
    kept = torch.zeros(0, dtype=torch.long)
    n_dropped = 0

    for group in idx_groups:
        group = torch.unique(group.to(torch.long).clamp(0, grid.n_tokens - 1))
        group = group[~seen[group]]          # 跨群組去重
        seen[group] = True
        room = n_budget - kept.numel()
        if group.numel() > room:
            n_dropped += group.numel() - room
            group = group[:room]
        kept = torch.cat([kept, group])

    # 空 ROI:保留一個 token。n_valid=0 會讓每一列都被遮蔽 -> softmax NaN。
    if kept.numel() == 0:
        kept = torch.zeros(1, dtype=torch.long)

    out = torch.zeros(n_budget, dtype=torch.long)
    out[: kept.numel()] = kept
    return out, int(kept.numel()), n_dropped


def _valid_mask(n_slots: int, n_valid: Tensor) -> Tensor:
    """[B] 的有效數 → [B, n_slots] 的布林旗標。

    仰賴 pack() 保證的「真索引緊密靠前」,所以 arange 比較就等於逐格旗標。
    見 cost_model.md 4.2。
    """
    ar = torch.arange(n_slots, device=n_valid.device)
    return ar[None, :] < n_valid[:, None]


def build_vit_masks(
    roi_token_idx: Tensor,
    n_valid: Tensor,
    dtype: torch.dtype = torch.float32,
    grid: Grid = SAM3_GRID,
) -> tuple[Tensor, Tensor]:
    """建 32 層 ViT 共用的兩份加法遮罩。

    規則(cost_model.md 5.2):
        windowed 可見(i,j) <=> window(i)==window(j) 且 valid(i) 且 valid(j)
        global   可見(i,j) <=> valid(i) 且 valid(j)
        兩者的對角線 i==j 恆可見 —— dummy 若整列被遮,softmax 會生 NaN,
        而遮罩是加法、救不了 NaN。

    Returns:
        (m_win, m_glb),各為 [B,1,N,N],可見處為 0、遮蔽處為 finfo(dtype).min。
    """
    idx = roi_token_idx if roi_token_idx.dim() == 2 else roi_token_idx[None]
    batch, n_slots = idx.shape
    device = idx.device

    valid = _valid_mask(n_slots, n_valid.to(device))            # [B,N]
    both = valid[:, :, None] & valid[:, None, :]                # [B,N,N]
    eye = torch.eye(n_slots, dtype=torch.bool, device=device)[None]

    wid = window_id(idx, grid)                                  # [B,N]
    same_window = wid[:, :, None] == wid[:, None, :]            # [B,N,N]

    neg = torch.finfo(dtype).min
    m_glb = torch.zeros(batch, 1, n_slots, n_slots, dtype=dtype, device=device)
    m_win = torch.zeros_like(m_glb)
    m_glb.masked_fill_(~(both | eye)[:, None], neg)
    m_win.masked_fill_(~((both & same_window) | eye)[:, None], neg)
    return m_win, m_glb


def dense_key_invalid(
    roi_token_idx: Tensor,
    n_valid: Tensor,
    grid: Grid = SAM3_GRID,
) -> Tensor:
    """[B,1,1,n_tokens] 布林:True = 這個 key 不在 ROI 裡、要遮掉。

    給 DETR encoder self-attn 與 decoder vision cross-attn 用。
    **key-only** 形狀是刻意的:完整的 [B,1,5184,5184] fp32 是 107 MB,會爆。

    用 scatter_add 計數而非 scatter 寫入:dummy 的索引都是 0,重複索引的
    寫入順序未定義,可能把真 token 0 的旗標覆蓋掉。
    """
    idx = roi_token_idx if roi_token_idx.dim() == 2 else roi_token_idx[None]
    batch, n_slots = idx.shape
    device = idx.device
    valid = _valid_mask(n_slots, n_valid.to(device)).to(torch.long)
    hits = torch.zeros(batch, grid.n_tokens, dtype=torch.long, device=device)
    hits.scatter_add_(1, idx, valid)
    return (hits == 0)[:, None, None, :]


def additive(key_invalid: Tensor, dtype: torch.dtype) -> Tensor:
    """布林遮罩 → 加法遮罩(可見 0、遮蔽 finfo(dtype).min)。"""
    return torch.zeros(key_invalid.shape, dtype=dtype,
                       device=key_invalid.device).masked_fill_(
        key_invalid, torch.finfo(dtype).min)


@dataclass
class RoiContext:
    """一幀的 ROI 全部衍生張量。由 make_context 建好後穿過整個模型。

    刻意是張量的容器而非行為物件:模型端只做 index_select 與加法,
    所有算術都留在 roi.py 以便單測。
    """

    token_idx: Tensor        # [B,N] int64
    n_valid: Tensor          # [B]   int64
    valid: Tensor            # [B,N] bool
    rope_idx_global: Tensor  # [N]   int64
    rope_idx_window: Tensor  # [N]   int64
    m_win: Tensor            # [B,1,N,N] 加法
    m_glb: Tensor            # [B,1,N,N] 加法
    key_invalid: Tensor      # [B,1,1,n_tokens] bool

    @property
    def n_slots(self) -> int:
        return self.token_idx.shape[1]

    def rope_idx(self, is_global: bool) -> Tensor:
        return self.rope_idx_global if is_global else self.rope_idx_window

    def vit_mask(self, is_global: bool) -> Tensor:
        return self.m_glb if is_global else self.m_win


def make_context(
    roi_token_idx: Tensor,
    n_valid: Tensor,
    dtype: torch.dtype,
    device,
    grid: Grid = SAM3_GRID,
) -> RoiContext:
    """把 pack() 的輸出擴成模型要的全部衍生張量。B=1 是唯一測試過的情形。

    三個 assert 守的是「pack() 保證但這裡不會重新推導」的前提,一旦有人繞過
    pack() 直接組 idx(例如階段 2 的 upstream_roi.py 把多個框的索引
    torch.cat 起來卻忘記 unique),要在這裡就大聲失敗,而不是讓
    scatter_add/attention 靜默算出錯的結果:

    - batch 必須是 1:B>1 時 torch.gather 用 [1,N,C] 索引配 [B,...] 輸入不會
      報錯,只會回傳 [1,N,C],下游 scatter_add 因此只寫 batch 0,其餘 batch
      靜默留零(無 NaN、無例外)。design.md 2.1 明確只承諾 B=1。
    - idx[:n_valid] 唯一:重複索引會讓 scatter_add 把對應 token 疊加、也會讓
      attention 把同一個 key 看成出現多次(等於加權)。
    - idx[:n_valid] 落在 [0, n_tokens):_valid_mask 的 arange 比較與
      index_select/gather 都假設索引合法;design.md 5.3 承諾非匯出模式下
      assert 越界索引,pack() 本身只 clamp,這裡是真正兌現承諾的地方。
    """
    idx = (roi_token_idx if roi_token_idx.dim() == 2 else roi_token_idx[None])
    idx = idx.to(device=device, dtype=torch.long)
    n_valid = n_valid.to(device=device, dtype=torch.long)

    assert idx.shape[0] == 1, (
        f"RoiContext 只支援 B=1(design.md 2.1),收到 batch={idx.shape[0]}"
    )
    n = int(n_valid[0].item())
    valid_prefix = idx[0, :n]
    assert int(torch.unique(valid_prefix).numel()) == n, (
        "roi_token_idx 的前 n_valid 個索引含有重複值——scatter_add 會把重複"
        "索引疊加、attention 也會把它當成同一個 key 出現多次。呼叫端(例如"
        "upstream_roi.py 手動 torch.cat 多組框的索引)必須先 torch.unique() "
        "再傳進來,不能只靠 make_context 事後發現。"
    )
    assert bool(((valid_prefix >= 0) & (valid_prefix < grid.n_tokens)).all()), (
        f"roi_token_idx 的前 n_valid 個索引必須落在 [0, {grid.n_tokens}) 之內"
        "(design.md 5.3:pack() 只 clamp,非匯出模式下這裡負責 assert 越界)。"
    )

    i_global, i_window = rope_indices(idx[0], grid)
    m_win, m_glb = build_vit_masks(idx, n_valid, dtype, grid)
    return RoiContext(
        token_idx=idx,
        n_valid=n_valid,
        valid=_valid_mask(idx.shape[1], n_valid),
        rope_idx_global=i_global,
        rope_idx_window=i_window,
        m_win=m_win,
        m_glb=m_glb,
        key_invalid=dense_key_invalid(idx, n_valid, grid),
    )
