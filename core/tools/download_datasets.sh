#!/usr/bin/env bash
# Скачать датасеты для обучения behavior_v1 в data/datasets/ (запускать из корня репозитория):
#     bash core/tools/download_datasets.sh            # KTH (бег/ходьба) + UR Fall (падения), ~760 МБ
#     bash core/tools/download_datasets.sh gmdcsa     # + GMDCSA-24 (падения), ещё 1,1 ГБ
# Уже скачанные файлы пропускаются — скрипт можно перезапускать после обрыва.
set -u
cd "$(dirname "$0")/../.."
D=data/datasets

echo "== KTH Actions: running, jogging, walking (~560 МБ)"
mkdir -p $D/kth
for a in running jogging walking; do
    if ls $D/kth/*_${a}_* >/dev/null 2>&1; then echo "   $a: уже есть"; continue; fi
    curl -L --fail -o $D/kth/$a.zip https://www.csc.kth.se/cvap/actions/$a.zip \
        && unzip -q -o $D/kth/$a.zip -d $D/kth && rm $D/kth/$a.zip
done
echo "   роликов KTH: $(ls $D/kth/*.avi 2>/dev/null | wc -l) (ожидается ~300)"

echo "== UR Fall: 30 падений x 2 камеры + 40 обычных действий (~200 МБ)"
mkdir -p $D/urfall
B=https://fenix.ur.edu.pl/~mkepski/ds/data
get() { [ -s "$D/urfall/$1" ] || curl -s --fail -o "$D/urfall/$1" "$B/$1" || echo "   не скачалось: $1"; }
get urfall-cam0-falls.csv
for i in $(seq -w 1 30); do get fall-$i-cam0.mp4; get fall-$i-cam1.mp4; printf "f%s " $i; done; echo
for i in $(seq -w 1 40); do get adl-$i-cam0.mp4; printf "a%s " $i; done; echo
echo "   роликов UR Fall: $(ls $D/urfall/*.mp4 2>/dev/null | wc -l) (ожидается 100)"

if [ "${1:-}" = "gmdcsa" ]; then
    echo "== GMDCSA-24 (1,1 ГБ)"
    mkdir -p $D/gmdcsa24
    curl -L --fail -C - -o $D/gmdcsa24/gmdcsa24.zip \
        "https://zenodo.org/records/13354453/files/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1.zip?download=1" \
        && unzip -q -o $D/gmdcsa24/gmdcsa24.zip -d $D/gmdcsa24 && rm $D/gmdcsa24/gmdcsa24.zip
fi
echo "== готово: $D"
