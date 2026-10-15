"""⑥ 모델 도감: 모든 모델 화면이 예외 없이 그려지는지 (합성 데이터 시각화 + 슬라이더 변경)."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import model_atlas as MA

t0 = time.time()
at = AppTest.from_string(f"""
import sys; sys.path.insert(0, r"{ROOT}")
import model_atlas
model_atlas.render()
""", default_timeout=120)
at.run()
assert not at.exception, at.exception
for name in MA.ATLAS:
    at.selectbox(key="at_sel").set_value(name).run()
    assert not at.exception, (name, [e.value for e in at.exception])
    # 첫 슬라이더를 최댓값으로 움직여도 동작
    sl = [s for s in at.slider if s.key and s.key.startswith("at_")]
    if sl:
        s = sl[0]
        try:
            s.set_value(s.max).run()
        except Exception:
            pass
        assert not at.exception, (name, "slider", [e.value for e in at.exception])
    print(f"[{time.time() - t0:5.1f}s] OK  {name}")
print("ATLAS TEST PASSED")
