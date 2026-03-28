# 통합됨: analyze_fde_epistemic_correlation.py 가 상관 분석 + 시나리오 시각화를 함께 수행합니다.
# 하위 호환: 이 파일은 동일 인자로 통합 스크립트를 호출합니다.
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    target = root / "analyze_fde_epistemic_correlation.py"
    argv = [sys.executable, str(target), "--only_scenario_viz"] + sys.argv[1:]
    raise SystemExit(subprocess.call(argv))
