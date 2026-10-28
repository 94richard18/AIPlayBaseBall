"""Batter training progress compilation (CPU only): existing clips + Chinese caption bars + cards."""
import sys
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
OUT = sys.argv[1] if len(sys.argv) > 1 else "videos/batter_progress_1028.mp4"
FONT = "C:/Windows/Fonts/msjh.ttc"
FONT_B = "C:/Windows/Fonts/msjhbd.ttc"
f_tag = ImageFont.truetype(FONT_B, 30)
f_cap = ImageFont.truetype(FONT, 28)
f_title = ImageFont.truetype(FONT_B, 54)
f_sub = ImageFont.truetype(FONT, 32)
f_card = ImageFont.truetype(FONT, 30)

# (file, start s, end s, date tag, caption lines)
SEGS = [
    ("videos/showcase_success.mp4", 0.0, 10.5, "10-06  第 1 階段：tee 打擊（超人類力矩）",
     ["模仿 OBP 揮棒動作 + 強化學習，先用強力致動器確認整條流程可行",
      "成果：16/16 揮棒全部擊中，飛行距離 158 m（目標 150 m）"]),
    ("videos/tee_perceptionB.mp4", 0.0, 9.0, "10-07  tee 位置隨機 + 感知雜訊",
     ["球的觀測加入 2 cm 偏差、1 cm 雜訊、100 ms 延遲，tee 位置每次隨機",
      "成果：每揮必中，平均飛行 84–97 m（這一球 146 m）"]),
    ("videos/batting_vs_pitches.mp4", 0.0, 7.0, "10-07  第一次面對投球（84 km/h 直球）",
     ["同一個打者直接打來球：碰得到球，但多為界外或揮空",
      "問題：打擊時機與擊球點都還沒學"]),
    ("videos/batter_human4_1021.mp4", 4.0, 11.5, "10-20  改為人類力量打者（Human3 / Human4）",
     ["力矩與關節速度改成人類等級，面對比賽球速 88–112 km/h",
      "成果：擊中率約 15% → 加 0.04 s 時機偏移後約 40%，初速約 20 m/s"]),
    ("videos/batter_human6_1021pm.mp4", 4.0, 11.0, "10-21  Human6：揮棒起動改用真實到達時間",
     ["揮棒起動 = 球到本壘時間 − 參考擊球時間 − 0.055 s（引擎同一規則）",
      "成果：擊中率 94.5%，但多數球仰角過低（中位 −34°）、落點約 1 m；這一球是少數飛出去的"]),
    ("videos/batter_h7_aim012_1025.mp4", 13.0, 20.0, "10-23  Human7：平滑仰角分數 + 瞄準點下修 0.12 m",
     ["仰角獎勵改成有梯度的版本，擊球點往下瞄 12 cm",
      "成果：仰角中位 +8°，擊中率 0.73，界內球開始飛出 15–30 m"]),
    ("videos/one_outs_h9b_1025.mp4", 0.0, 7.0, "10-25  目標改為 ONE OUTS（一打席、無野手）",
     ["Human9 以「界內 29 m」為目標再訓練；比賽引擎用實測擊球表打 1000 場",
      "成果：打者勝率 13% → 交付版（力量 1.5 倍）17.6%，外野球 3.5%"]),
    ("videos/batter_h9b_12500_e1.5_p3c_1027pm.mp4", 12.0, 18.0, "10-27  面對 RL 投手 P3c 的真實投球（約 64 km/h）",
     ["投手改為「控球優先」後球速變慢，打者對慢球的時機要重新對齊",
      "成果：好球帶擊中率 0.46（單獨訓練環境、同投球資料）"]),
    ("videos/m3_s3_k12.mp4", 0.0, 11.4, "10-28  真正同場模擬：投手與打者同一個場景、同一顆球",
     ["兩個策略同時即時運作，球由投手手中真的飛到打者的棒上",
      "成果：好球帶擊中率 0.515（與單獨環境一致），擊中初速中位 16 m/s"]),
]

TITLE = ("AIB-1 打者訓練歷程", "2026-10-05 → 10-28　從 tee 打擊到與 RL 投手同場對決")
END = [
    "現況（10-28）",
    "・ONE OUTS 打者勝率 9.8%（20000 次模擬打席，幾乎都是四壞）",
    "・好球帶擊中率 0.515、擊中後跌倒 49%、尚無外野球（初速中位 16 m/s，外野需約 25）",
    "",
    "進行中",
    "・不揮棒時站穩（看球不跌倒）　・打者身後的牆與好球帶　・打者力量方向待決定",
    "・最終影片：完整打席 + 球數卡，11-26 交付",
]


def fit(frame):
    im = Image.fromarray(frame).convert("RGB")
    w, h = im.size
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (12, 14, 18))
    canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))
    return canvas


def caption(im, tag, lines, alpha=1.0):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    bar_h = 40 + 36 * len(lines) + 16
    d.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, int(165 * alpha)))
    d.rectangle([0, 0, W, 50], fill=(0, 0, 0, int(150 * alpha)))
    d.text((18, 8), tag, font=f_tag, fill=(255, 220, 90, int(255 * alpha)))
    y = H - bar_h + 14
    for ln in lines:
        d.text((22, y), ln, font=f_cap, fill=(255, 255, 255, int(255 * alpha)))
        y += 36
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def card(lines, secs, big_first=False):
    im = Image.new("RGB", (W, H), (16, 20, 26))
    d = ImageDraw.Draw(im)
    if big_first:
        d.text((W // 2, 280), lines[0], font=f_title, fill=(255, 255, 255), anchor="mm")
        d.text((W // 2, 370), lines[1], font=f_sub, fill=(200, 210, 220), anchor="mm")
    else:
        y = 120
        for ln in lines:
            f = f_tag if ln and not ln.startswith("・") else f_card
            col = (255, 220, 90) if f is f_tag else (235, 235, 235)
            d.text((90, y), ln, font=f, fill=col)
            y += 52 if ln else 26
    arr = np.asarray(im)
    return [arr] * int(secs * FPS)


def main():
    wr = iio.get_writer(OUT, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                        output_params=["-profile:v", "high", "-crf", "20", "-preset", "medium"],
                        macro_block_size=16)
    n = 0
    for fr in card(list(TITLE), 3.5, big_first=True):
        wr.append_data(fr); n += 1
    for path, t0, t1, tag, lines in SEGS:
        rd = iio.get_reader(path)
        meta = rd.get_meta_data()
        fps = meta.get("fps", 30)
        i0, i1 = int(t0 * fps), int(t1 * fps)
        k = 0
        for i, frame in enumerate(rd):
            if i < i0:
                continue
            if i >= i1:
                break
            if fps > FPS + 1 and (i % 2):  # 60 fps sources -> drop every other frame
                continue
            im = fit(frame)
            fade = min(1.0, k / 8.0)
            im = caption(im, tag, lines, alpha=fade)
            wr.append_data(np.asarray(im)); n += 1; k += 1
        rd.close()
        print(f"{path}: {k} frames", flush=True)
    for fr in card(END, 8.0):
        wr.append_data(fr); n += 1
    wr.close()
    print(f"wrote {OUT}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
