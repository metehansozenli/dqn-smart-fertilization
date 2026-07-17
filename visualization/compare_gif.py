"""
DQN vs Random GIF — Turkish labels, clear growth, health via color.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import numpy as np
from PIL import Image
from env.agriculture_env import AgricultureEnv
from agent.dqn_agent import DQNAgent

WIDTH, HEIGHT = 1100, 920
HALF = HEIGHT // 2


class CompareGIF:
    def __init__(self, seed=2):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.ft = pygame.font.SysFont("DejaVu Sans", 20, bold=True)
        self.fm = pygame.font.SysFont("DejaVu Sans", 15, bold=True)
        self.fs = pygame.font.SysFont("DejaVu Sans", 13)
        self.fx = pygame.font.SysFont("DejaVu Sans", 12)

        self.env_dqn = AgricultureEnv(use_real_data=True, max_days=30, seed=seed)
        self.env_rnd = AgricultureEnv(use_real_data=True, max_days=30, seed=seed)
        self.agent = DQNAgent(9, self.env_dqn.n_actions, device="cpu")
        mp = "results/hybrid_dqn_model.pth"
        if os.path.exists(mp):
            self.agent.load(mp)
            self.agent.epsilon = 0.0
            print("Model yuklendi")

        self.plants_top = self._grid(y0=HALF - 48)
        self.plants_bot = self._grid(y0=HEIGHT - 48)

    def _grid(self, y0):
        pts = []
        sx, sy = 100, 36
        for row in range(4):
            for col in range(10):
                jx = float(np.sin(row * 2 + col) * 4)
                pts.append({"x": 55 + col * sx + jx, "y": y0 - (3 - row) * sy})
        return pts

    def plant_color(self, h):
        # health only affects COLOR (green -> yellow -> brown)
        if h < 0.15:
            return (90, 55, 30)
        if h < 0.35:
            return (140, 100, 35)
        if h < 0.55:
            return (120, 145, 40)
        if h < 0.75:
            return (70, 160, 50)
        return (35, 175, 55)

    def draw_plant(self, x, y, size, h):
        # size from growth (0..1); color from health
        color = self.plant_color(h)
        stem_h = int(8 + size * 58)
        leaf_w = int(6 + size * 22)
        leaf_h = int(5 + size * 13)
        stem_w = 2 + int(size * 3)
        pygame.draw.ellipse(self.screen, (60, 42, 26),
                            (x - leaf_w // 2 - 1, y + 2, leaf_w + 6, 7))
        stem_col = (40, 95, 32) if h > 0.4 else (95, 70, 35)
        pygame.draw.rect(self.screen, stem_col,
                         (x - stem_w // 2, y - stem_h + 5, stem_w, stem_h))
        n = 2 + int(size * 6)
        offs = [(-1, -0.2), (1, -0.18), (0, -0.55), (-0.65, -0.4),
                (0.65, -0.38), (-0.35, -0.75), (0.35, -0.72), (0, -0.9)]
        for i in range(n):
            dx, dy = offs[i % len(offs)]
            lx = x + int(dx * leaf_w * 0.9)
            ly = y - int(stem_h * (0.35 + abs(dy) * 0.45))
            c = color if i < 5 else tuple(max(0, v - 22) for v in color)
            pygame.draw.ellipse(self.screen, c,
                                (lx - leaf_w // 2, ly - leaf_h // 2, leaf_w, leaf_h))
        if size > 0.15:
            cw = int(leaf_w * (1.05 + 0.2 * size))
            ch = int(leaf_h * (1.0 + 0.15 * size))
            pygame.draw.ellipse(self.screen, color,
                                (x - cw // 2, y - stem_h - ch // 4, cw, ch))

    def draw_soil(self, y0, h, wet=False, fert=False):
        base = (105, 72, 44)
        if wet:
            base = (70, 50, 34)
        if fert:
            base = (122, 92, 38)
        pygame.draw.rect(self.screen, base, (0, y0, WIDTH, h))
        for i in range(0, WIDTH, 22):
            pygame.draw.line(self.screen, (82, 55, 32), (i, y0), (i + 40, y0 + h), 2)

    def draw_water(self, plants, alpha):
        for p in plants:
            x, y = p["x"], p["y"]
            s = pygame.Surface((28, 11), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (40, 70, 110, int(90 * alpha)), (0, 0, 28, 11))
            self.screen.blit(s, (x - 14, y + 2))

    def draw_fert(self, plants, alpha):
        for p in plants:
            x, y = p["x"], p["y"]
            s = pygame.Surface((30, 10), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (180, 150, 40, int(120 * alpha)), (0, 0, 30, 10))
            self.screen.blit(s, (x - 15, y + 2))
            rng = np.random.RandomState(int(x + y) % 999)
            for _ in range(5):
                pygame.draw.circle(self.screen, (210, 190, 45),
                                   (int(x + rng.randint(-11, 11)),
                                    int(y + rng.randint(2, 9))), 2)

    def draw_sky(self, y0, h):
        for y in range(h):
            t = y / max(h, 1)
            pygame.draw.line(self.screen,
                             (int(45 + t * 70), int(90 + t * 85), int(150 + t * 60)),
                             (0, y0 + y), (WIDTH, y0 + y))

    def reason(self, moisture, nutrient, fert, water):
        if fert > 0:
            return f"Besin={nutrient:.0f} → gubrele +{fert} kg"
        if water > 0:
            return f"Nem={moisture:.0f}% → sulama"
        return f"Nem={moisture:.0f}% Besin={nutrient:.0f} → bekle"

    def panel(self, y0, title, color, day, info, state, act, why, plants, effect, alpha):
        soil_y = y0 + HALF - 175
        self.draw_sky(y0 + 52, soil_y - (y0 + 52))
        wet = effect == "water" and alpha > 0.05
        fert = effect == "fert" and alpha > 0.05
        self.draw_soil(soil_y, y0 + HALF - soil_y, wet=wet, fert=fert)
        if fert:
            self.draw_fert(plants, alpha)
        if wet:
            self.draw_water(plants, alpha)

        growth = float(info["cumulative_growth"])
        health = float(info["health"])
        tox = float(info.get("toxicity", state[8]))
        yld = float(info["current_yield"])
        tf = float(info["total_fert_used"])
        moisture = float(state[1])
        nutrient = float(state[2])

        # SIZE from growth only (clear growth over season)
        # floor so seedlings are visible; scale so end is clearly taller
        size = float(np.clip((growth - 1.0) / 0.55, 0.08, 1.0))
        # random with dead health: slightly smaller but still growth-based, color shows damage
        if health < 5:
            size *= 0.45  # collapsed plants when fully dead
        h_n = float(np.clip(health / 100.0, 0, 1))
        for p in plants:
            self.draw_plant(p["x"], p["y"], size, h_n)

        bar = pygame.Surface((WIDTH, 26), pygame.SRCALPHA)
        bar.fill((*color, 235))
        self.screen.blit(bar, (0, y0))
        self.screen.blit(self.fm.render(title, True, (255, 255, 255)), (10, y0 + 4))
        self.screen.blit(self.fs.render(f"Gun {day:02d}/30", True, (255, 255, 255)), (340, y0 + 5))

        ms = pygame.Surface((WIDTH, 26), pygame.SRCALPHA)
        ms.fill((10, 14, 22, 230))
        self.screen.blit(ms, (0, y0 + 26))
        metrics = (f"Verim={yld:,.0f} kg/ha   Saglik={health:.0f}   Toksisite={tox:.1f}   "
                   f"Buyume=x{growth:.2f}   GubreToplam={tf:.0f} kg   "
                   f"Nem={moisture:.0f}%   Besin={nutrient:.0f}")
        self.screen.blit(self.fx.render(metrics, True, (255, 240, 160)), (10, y0 + 31))

        fk, w = act["fert_kg"], act["water"]
        if fk > 0:
            dbg, dtxt = (100, 85, 15), f"AKSIYON: GUBRELE +{fk} kg/ha"
        elif w > 0:
            dbg, dtxt = (20, 55, 100), f"AKSIYON: SULA +{w} mm"
        else:
            dbg, dtxt = (50, 50, 55), "AKSIYON: BEKLE"
        db = pygame.Surface((WIDTH, 34), pygame.SRCALPHA)
        db.fill((*dbg, 245))
        self.screen.blit(db, (0, y0 + HALF - 34))
        self.screen.blit(self.fs.render(dtxt, True, (255, 255, 255)), (10, y0 + HALF - 30))
        self.screen.blit(self.fx.render(f"Gerekce: {why}", True, (235, 235, 200)),
                         (10, y0 + HALF - 14))

    def capture(self):
        data = pygame.image.tostring(self.screen, "RGB")
        return Image.frombytes("RGB", (WIDTH, HEIGHT), data)

    def generate(self, out_path="results/compare_dqn_vs_random.gif"):
        frames = []
        s_d, _ = self.env_dqn.reset()
        s_r, _ = self.env_rnd.reset()
        self.env_rnd.base_yield = self.env_dqn.base_yield
        self.env_rnd.state = self.env_dqn.state.copy()
        s_r = self.env_rnd.state.copy()

        print("GIF uretiliyor (TR, buyume net)...")
        fy_d = fy_r = ff_d = ff_r = 0

        for day in range(1, 31):
            m_d, n_d = float(s_d[1]), float(s_d[2])
            m_r, n_r = float(s_r[1]), float(s_r[2])

            a_d = self.agent.select_action(s_d, evaluate=True)
            a_r = int(np.random.randint(0, self.env_rnd.n_actions))
            act_d = self.env_dqn.actions[a_d]
            act_r = self.env_rnd.actions[a_r]
            why_d = self.reason(m_d, n_d, act_d["fert_kg"], act_d["water"])
            why_r = self.reason(m_r, n_r, act_r["fert_kg"], act_r["water"])

            if act_d["fert_kg"] > 0:
                ed, efd = "fert", 9
            elif act_d["water"] > 0:
                ed, efd = "water", 6
            else:
                ed, efd = None, 0
            if act_r["fert_kg"] > 0:
                er, efr = "fert", 9
            elif act_r["water"] > 0:
                er, efr = "water", 6
            else:
                er, efr = None, 0

            s_d, _, done_d, _, info_d = self.env_dqn.step(a_d)
            s_r, _, done_r, _, info_r = self.env_rnd.step(a_r)
            fy_d, fy_r = info_d["current_yield"], info_r["current_yield"]
            ff_d, ff_r = info_d["total_fert_used"], info_r["total_fert_used"]

            for _ in range(5):
                ad = min(1.0, efd / 5.0) if efd > 0 else 0
                ar = min(1.0, efr / 5.0) if efr > 0 else 0
                if efd > 0:
                    efd -= 1
                if efr > 0:
                    efr -= 1
                self.screen.fill((18, 22, 28))
                self.panel(0, "DQN AJAN (egitilmis politika)", (0, 115, 85),
                           day, info_d, s_d, act_d, why_d, self.plants_top, ed, ad)
                self.panel(HALF, "RASTGELE POLITIKA (ogrenme yok)", (145, 45, 40),
                           day, info_r, s_r, act_r, why_r, self.plants_bot, er, ar)
                pygame.draw.line(self.screen, (230, 230, 230), (0, HALF), (WIDTH, HALF), 3)
                frames.append(self.capture())

            if day % 5 == 0 or act_d["fert_kg"] > 0:
                print(f"Gun {day:02d} | DQN V={fy_d:.0f} S={info_d['health']:.0f} T={info_d['toxicity']:.1f} "
                      f"|| RND V={fy_r:.0f} S={info_r['health']:.0f} T={info_r['toxicity']:.1f}")

            if done_d or done_r:
                break

        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((8, 12, 20, 205))
        self.screen.blit(ov, (0, 0))
        box = pygame.Surface((740, 300), pygame.SRCALPHA)
        box.fill((18, 28, 40, 250))
        self.screen.blit(box, (WIDTH // 2 - 370, HEIGHT // 2 - 150))
        self.screen.blit(self.ft.render("GERCEK MODEL CIKTISI  |  DQN vs RASTGELE", True, (0, 220, 180)),
                         (WIDTH // 2 - 240, HEIGHT // 2 - 130))
        lines = [
            f"DQN:      Verim {fy_d:,.0f} kg/ha   |   Gubre {ff_d:.0f} kg/ha",
            f"Rastgele: Verim {fy_r:,.0f} kg/ha   |   Gubre {ff_r:.0f} kg/ha",
            "",
            "Tum sayilar ortam ve egitilmis DQN modelinden gelir.",
            "Bitki boyu = buyume (G). Renk = saglik (toksisite sari/kahve yapar).",
            "Ayni taban verim; sadece karar politikasi farklidir.",
        ]
        y = HEIGHT // 2 - 80
        for ln in lines:
            self.screen.blit(self.fs.render(ln, True, (235, 235, 235)), (WIDTH // 2 - 340, y))
            y += 30
        for _ in range(26):
            frames.append(self.capture())

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=115, loop=0, optimize=True)
        print(f"Kaydedildi -> {out_path} ({len(frames)} kare)")
        print(f"SONUC  DQN V={fy_d:.0f} G={ff_d:.0f}  |  RND V={fy_r:.0f} G={ff_r:.0f}")
        pygame.quit()
        return out_path


if __name__ == "__main__":
    np.random.seed(0)
    CompareGIF(seed=2).generate()
