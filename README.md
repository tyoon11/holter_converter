
# Holter ECG Raw to HDF5 변환 파이프라인

이 프로젝트는 Holter ECG raw 데이터를 분석 가능한 구조로 변환하여 `.h5` 형식으로 저장하는 파이프라인을 제공합니다.

---

## 📦 의존성 설치

다음 Python 패키지를 설치해야 합니다:

```bash
pip install pandas numpy h5py matplotlib neurokit2 dtw wfdb
```

---

## 📁 파일 구성

| 파일명                      | 설명                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `generate_valid_list.py` | `.hea`, `.SIG`, `.ANN`, `.json` 네 가지 파일이 모두 존재하는 valid 레코드를 탐색하여 `valid_records.csv`로 저장합니다.                            |
| `utils.py`               | HDF5 변환에 필요한 유틸리티 함수들을 정의한 모듈입니다. (신호 품질, fiducial 포인트 추출 등 포함)                                                         |
| `create_h5_structure.py` | ECG 세그먼트 기반의 HDF5 저장 구조를 정의하고, 데이터를 저장하는 함수들을 포함합니다.                                                                    |
| `convert_to_h5.py`       | `valid_records.csv`를 기준으로 raw 데이터를 `.h5` 형식으로 변환합니다. `ray`를 사용하여 병렬 처리되며, CPU 개수는 `ray.init(num_cpus=...)`로 조절할 수 있습니다. |
| `h5_test.ipynb`          | 생성된 `.h5` 파일의 구조를 탐색하고 내용을 시각화하는 Jupyter Notebook입니다.                                                                   |

---

## 🚀 실행 흐름

1. `generate_valid_list.py` 실행
   → 유효한 레코드 목록(`valid_records.csv`) 생성

   ```bash
   python generate_valid_list.py
   ```

2. `convert_to_h5.py` 실행
   → raw 데이터를 HDF5 형식으로 병렬 변환

   ```bash
   python convert_to_h5.py
   ```

   ⚠️ 병렬 처리에 사용할 CPU 개수는 `ray.init(num_cpus=...)`에서 조절하세요.

---

## 🧪 HDF5 구조 확인

`h5_test.ipynb`를 통해 HDF5 구조와 데이터 내용을 탐색하고 시각화할 수 있습니다.


