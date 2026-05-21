# MediBoxLesion

MediBoxLesion, sınırlayıcı kutu destekli aday lezyon analizi için geliştirilmiş bir tıbbi görüntüleme prototipidir. Proje, MELA CT hacimleri üzerinde çalışır ve CT kesitleri, anotasyon sınırlayıcı kutuları, pseudo-mask çıktıları ve model tahmin maskelerini karşılaştırmak için Streamlit tabanlı bir arayüz sunar.

> Bu proje araştırma ve staj prototipi niteliğindedir. Klinik tanı veya nihai segmentasyon aracı olarak kullanılmamalıdır.

## Proje Kapsamı

- MELA NIfTI CT hacimlerini yükleme ve görselleştirme.
- Anotasyon sınırlayıcı kutularını ve ayarlanabilir ROI bölgelerini gösterme.
- Pseudo-mask çıktıları ile model tahmin maskelerini karşılaştırma.
- Eşikleme, en büyük bileşen seçimi, morfolojik temizlik ve ROI kısıtı ile isteğe bağlı maske iyileştirme.
- Seçilen vakalar için temel kullanıcı geri bildirimi toplama.
- Arayüz üzerinden vaka bazlı CSV raporu dışa aktarma.
- Bağımsız scriptler ile yükleme, bellek, inference ve arayüz performansını ölçme.

## Proje Yapısı

```text
app/
  streamlit_app.py              Ana Streamlit arayüz uygulaması
  utils/mask_refinement.py      Tahmin iyileştirme yardımcı fonksiyonları
data/
  mela/                         MELA görüntüleri, maskeleri ve anotasyon tabloları
  nsclc/                        NSCLC görüntü ve maske dizileri
docs/                           Haftalık raporlar ve final teslim notları
models/                         Eğitilmiş model checkpoint dosyaları
notebooks/
  mela/                         MELA ön işleme, pseudo-mask ve inference notebook'ları
  nsclc/                        NSCLC keşif, veri seti ve eğitim notebook'ları
results/
  mela/                         Batch tahminler ve inference özetleri
  performance/                  Benchmark sonuç CSV dosyaları
scripts/                        Benchmark scriptleri
requirements.txt                Python bağımlılıkları
```

## Ana Uygulama

Ana arayüz dosyası:

```bash
streamlit run app/streamlit_app.py
```

Uygulamanın beklediği temel dosyalar:

- `data/mela/annotations/mela_master_dataframe.csv`
- `results/mela/mela_batch_inference_summary.csv`
- `data/mela/images/{train,val}/*.nii.gz`
- `data/mela/masks/{train,val}/*_mask.nii.gz`
- `results/mela/mela_batch_predictions/*_pred.npy`

## Kurulum

Sanal ortam oluşturup bağımlılıkları kurun:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Arayüzü çalıştırın:

```bash
streamlit run app/streamlit_app.py
```

## Arayüz Özellikleri

- Tahmin tipi filtresi: `all`, `empty`, `small`, `medium`, `large`.
- Vaka metrikleri: tahmin hacmi, tahmin tipi, kesit sayısı, veri seti bölümü.
- Dört panelli karşılaştırma: CT, sınırlayıcı kutulu CT, pseudo-mask ve tahmin/iyileştirilmiş tahmin.
- Manuel veya otomatik en iyi kesit seçimi.
- Maske görünen kesit galerisi.
- İsteğe bağlı iyileştirme kontrolleri.
- Vaka bazlı CSV rapor indirme.
- Kullanıcı geri bildirimini `results/user_feedback.csv` dosyasına kaydetme.

## Veri Özeti

Mevcut yerel veri özeti:

- MELA anotasyon satırı: 884
- Batch inference satırı: 370
- Batch tahmin dosyası: 370
- MELA train görüntüsü: 260
- MELA validation görüntüsü: 110
- MELA train maskesi: 260
- MELA validation maskesi: 110
- NSCLC görüntü/maske çifti: 106

Mevcut batch özetindeki tahmin tipi dağılımı:

| Tip | Adet |
| --- | ---: |
| small | 92 |
| medium | 268 |
| empty | 7 |
| large | 3 |

## Performans Scriptleri

Benchmark scriptleri proje kökünü otomatik olarak çözer. Bu nedenle proje klasörü taşınsa bile çalıştırılabilir.

```bash
python scripts/benchmark_loading.py
python scripts/benchmark_memory.py
python scripts/benchmark_inference.py
python scripts/benchmark_ui.py
```

Çıktılar `results/performance/` klasörüne yazılır.

## Final Teslim Notları

Önerilen final teslim içeriği:

- Bu README dosyasını içeren final GitHub reposu.
- `docs/` altında güncel dokümantasyon.
- `docs/FINAL_REPORT_PHASE2.md` temel alınarak hazırlanmış final raporu.
- `docs/FINAL_PRESENTATION_OUTLINE.md` temel alınarak hazırlanmış sunum.
- `docs/DEMO_VIDEO_SCRIPT.md` akışına göre çekilmiş demo videosu.
- Final toplantısından önce test edilmiş canlı demo.

## Bilinen Sınırlılıklar

- Arayüz bir prototip görüntüleme aracıdır ve klinik doğrulama sağlamaz.
- Model eğitim kodları ağırlıklı olarak notebook yapısında korunmuştur; tek parça üretim eğitim paketi değildir.
- Büyük tıbbi görüntü, maske, tahmin ve model dosyaları herkese açık GitHub tesliminde Git LFS veya harici depolama gerektirebilir.
- `.venv` farklı makineler arasında taşınabilir olmayabileceği için Python ortamı teslim makinesinde yeniden kurulmalıdır.
