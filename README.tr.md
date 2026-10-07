# Catalog Organizer

**Mücevher CAD dosyaları (`.3dm` / `.stl`) için yerel çalışan katalog aracı.**
Arşivi tarar, her parçayı ölçer, görüntü modeliyle sınıflandırır, fiyatlar, Etsy / WooCommerce
ilanlarını yazar ve canlı bir 3D görüntüleyicide gösterir. Aynı görüntüleyici PDF kataloğunun
resimlerini de çizer.

[![Lisans: GPL v3+](https://img.shields.io/badge/lisans-GPL--3.0--or--later-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![Durum](https://img.shields.io/badge/durum-beta-orange.svg)

🇬🇧 **English:** [README.md](README.md) · 📖 **Talimatname:** [docs/KULLANIM_KILAVUZU.md](docs/KULLANIM_KILAVUZU.md)

![Render sekmesi: aynı parça katalogdaki iki görünümden, ikisi de canlı oynatıcı](docs/images/render-tab.png)

*Render sekmesi, projeyle gelen örnek yüzük: katalog sayfasının bastığı iki görünüm yan yana, ikisi
de canlı 3D oynatıcı. Taş, yüzüğün boş yuvasına otomatik yerleştirildi.*

---

## Ne yapar

| | |
|---|---|
| **Analyze** | Yalnızca geometri, yapay zekâ gerekmez. Ölçüler, yüzük iç çapı, sekiz alaşım için metal ağırlığı (net, koçan, final), taş boyutu / adedi / karatı, koçan tespiti. Tek dosya veya klasör; CSV'ye aktarır. |
| **Process** | Bir klasördeki tüm CAD dosyalarında tam işlem hattı: görüntüler → model ile sınıflandırma ve etiket → ölçüm → katalog kaydı. Duraklat, devam et, iptal; CSV yazar. |
| **Search** | Kategori, alt kategori, taş durumu, marka, koçan filtreleri; etiket ve açıklamalarda serbest metin. Bellekte çalışır, onbinlerce kayıtta anında. |
| **Publish** | Seçim listeleri, ilan metni üretimi, kendi formüllerinle fiyat, **Etsy** ve **WooCommerce** CSV'si, **PDF katalog** (sayfada iki görünüm). |
| **Render** | Canlı 3D oynatıcı (gömülü Chromium + three.js). Maden ve taş rengini anında dene, serbestçe döndür, beğendiğin açıyı katalog görünümü olarak kaydet. PDF resimlerini de aynı motor çizer. |
| **Diagnostics / Settings / Logs** | Her aşamayı tek tek çalıştır ve süresini gör; model sağlayıcısını seç, fiyatları düzenle; denetim kaydını izle. |

Yüz binlerce CAD dosyası olan bir atölye için: bir şeyi satabilmeden önce *arşivde ne olduğunu*
bilmek gerekir.

## Öne çıkanlar

* **Varsayılan olarak yerel.** Varsayılan görüntü modeli [Ollama](https://ollama.com) ile kendi
  bilgisayarında çalışır. Bulut sağlayıcıları (OpenAI, MiniMax, GLM, OpenCode Zen, OpenAI uyumlu
  herhangi bir uç nokta) isteğe bağlıdır; anahtarlar dosyada değil, işletim sisteminin kimlik
  bilgisi deposunda durur. Telemetri yoktur. Görüntüler yalnızca seçtiğin sağlayıcıya gider.
* **Ekran görüntüsü değil, gerçek bir render motoru.** Taşlar kendi faset düzlemlerine karşı
  analitik olarak ışın izlenir (dispersiyon ve renge göre soğurma ile), metale köşe başına ortam
  tıkanması pişirilir, her şey bir kez tonlanır. Ayrıntı: [docs/RENDER_ENGINE.md](docs/RENDER_ENGINE.md).
* **Tutarlı fotoğraf.** Bir kategorinin tüm ürünleri aynı açıdan çekilir; yüzükler başı gösterecek
  şekilde, düz parçalar kameraya dönük. Elle seçilen açı ürün başına saklanır.
* **Mantığı anlaşılır ağırlık.** Boolean birleşim hacmi (üst üste binen gövdeler iki kez sayılmaz),
  Rhino varsa kesin kesici çıkarma, yoksa belgelenmiş bir sezgisel yöntem.
* **STL konusunda dürüstlük.** STL, makineye girmeden önceki parçadır: taş yoktur, yalnızca taşlar
  için açılmış yuvalar vardır. Program yuvaları bulup taşları yerleştirebilir, ama yalnızca sizin
  (veya katalog kaydının) taşlı dediği ürünlerde ve sonucu tahmin olarak etiketler. `.3dm` taşları
  bir katmanda taşır, tahmine gerek kalmaz. Ayrıntı: [kılavuz](docs/KULLANIM_KILAVUZU.md#9-stl-mi-3dm-mi).

## Hızlı başlangıç (Windows)

**Python 3.11 veya 3.12** ve **WebGL 2** destekli bir ekran kartı gerekir (yeni hemen her kart).

```powershell
git clone https://github.com/emrekaandrms/catalog-organizer.git
cd catalog-organizer
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m catalog_organizer.app
```

Kendi CAD dosyan olmadan dene: **Render** sekmesinde **Dosyadan aç…** düğmesine bas ve
[`examples/demo_ring.stl`](examples/README.md) dosyasını seç (telifsiz, kodla üretilmiş bir yüzük).
**Taşları yuvalara yerleştir** kutusunu işaretle, boş yuvaya taş girer.

Yapay zekâ ile sınıflandırma için [Ollama](https://ollama.com) kur ve varsayılan modeli indir:

```powershell
ollama pull qwen3.5:9b-q4_K_M
```

Ayrıntılı kurulum, isteğe bağlı Rhino desteği ve sorun giderme: [docs/INSTALL.md](docs/INSTALL.md).

## Komut satırı

```powershell
python -m catalog_organizer.app                                   # arayüz
python -m catalog_organizer.app analyze DOSYA [DOSYA ...] --csv cikti.csv   # yalnızca geometri
python -m catalog_organizer.app analyze --folder KLASOR --csv cikti.csv
python -m catalog_organizer.app run-pilot --n 5 --roots KLASOR     # birkaç dosyada başsız işlem hattı
python -m catalog_organizer.app validate --truth gercek.csv        # tahminleri gerçek değerlerle karşılaştır
```

## Sınırlar (dürüst liste)

* **Windows 11**'de geliştirildi ve test edildi. Diğer platformlar denenmedi.
* Ağırlık, taş ve karat değerleri az sayıda gerçek dosyada ayarlanmış **tahminlerdir**. Fiyatlamadan
  önce terazi ile doğrula.
* STL yuvalarına taş yerleştirme sihir değil, bir dedektör: yuvarlak, daralan, üstü açık yuvaları
  bulur; pavé kanalları ve delikli bantları görmez.
* Render motoru WebGL 2'li bir ekran kartı ister. Betik ve testler için farklı görünümlü bir VTK
  yedeği var.

## Lisans

**GPL-3.0-or-later**, bkz. [LICENSE](LICENSE). Arayüz PyQt6 üzerine kurulu; PyQt6 GPL v3 (veya
Riverbank'tan ticari lisans) olduğundan bu proje de GPL. Üçüncü taraf yazılımlar:
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

Etsy, WooCommerce, Rhino, MatrixGold, Qt ve diğer adlar sahiplerinin ticari markalarıdır. Bu
proje bağımsızdır; adı geçen firmalarla bağlantısı yoktur, onlarca onaylanmamıştır.
