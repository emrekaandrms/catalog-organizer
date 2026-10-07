# Catalog Organizer — Kullanım Kılavuzu (Talimatname)

Bu kılavuz, programı ilk kez kuran birinin sıfırdan çalışır duruma gelmesi ve günlük işini
yapabilmesi için yazıldı. İngilizce özet: [USER_GUIDE.md](USER_GUIDE.md).

**İçindekiler**

1. [Program ne yapar, ne yapmaz](#1-program-ne-yapar-ne-yapmaz)
2. [Gereksinimler](#2-gereksinimler)
3. [Kurulum](#3-kurulum)
4. [İlk açılış ve model seçimi](#4-ilk-açılış-ve-model-seçimi)
5. [Arayüze genel bakış](#5-arayüze-genel-bakış)
6. [İş akışları](#6-iş-akışları)
7. [Render sekmesi ve PDF katalog](#7-render-sekmesi-ve-pdf-katalog)
8. [Ayarlar ve yapılandırma dosyaları](#8-ayarlar-ve-yapılandırma-dosyaları)
9. [STL mi, 3DM mi?](#9-stl-mi-3dm-mi)
10. [Verilerin yeri, yedekleme, sıfırlama](#10-verilerin-yeri-yedekleme-sıfırlama)
11. [Sorun giderme](#11-sorun-giderme)
12. [Doğruluk ve sınırlar](#12-doğruluk-ve-sınırlar)

---

## 1. Program ne yapar, ne yapmaz

**Yapar**

* Bir klasördeki (alt klasörler dahil) `.3dm` ve `.stl` mücevher CAD dosyalarını bulur.
* Her dosyadan **ölçü** (boyutlar, yüzük iç çapı), **metal ağırlığı** (sekiz alaşım için net, koçan
  ve final), **taş bilgisi** (boyut, adet, karat) ve **koçan (döküm sapı)** tespiti çıkarır.
  Bunların hiçbiri için yapay zekâ gerekmez.
* Dosyadan görüntüler üretir ve bir **görüntü modeline** (VLM) göstererek **kategori, alt kategori,
  etiketler, marka izi ve açıklama** çıkarır.
* Her şeyi bir **kataloğa** yazar; kataloğu aratır, listeler halinde seçersin.
* Seçili ürünler için **Etsy / WooCommerce ilan metni** yazar, **fiyat** hesaplar, **CSV** verir.
* Parçayı **canlı 3D** gösterir; maden ve taş rengini dener; **PDF katalog** üretir.

**Yapmaz**

* CAD dosyalarını değiştirmez, taşımaz, silmez. Dosyalar yalnızca okunur.
* Mesh onarımı, üretim ya da teraziye koyma yerine geçmez: ağırlıklar **tahmindir**.
* İlanları Etsy / WooCommerce'e kendisi yüklemez; CSV üretir, yüklemeyi sen yaparsın.

## 2. Gereksinimler

| | |
|---|---|
| İşletim sistemi | **Windows 10/11** (geliştirildi ve test edildi: Windows 11) |
| Python | **3.11 veya 3.12** |
| Ekran kartı | **WebGL 2** destekli (yeni hemen her kart). Render motoru ekran kartında çizer. |
| Bellek | 16 GB önerilir (büyük dosyalar milyonlarca üçgen olabilir) |
| Disk | Önbellek için boş yer; program 50 GB altında uyarır |
| Yapay zekâ (isteğe bağlı) | [Ollama](https://ollama.com) (yerel) **veya** bulut sağlayıcı anahtarı |
| Rhino (isteğe bağlı) | Kendi lisanslı Rhino 5–8 kurulumun; yalnızca kesin kesici hacmi için |

Yapay zekâ olmadan da **Analyze** sekmesi, **Render** sekmesi ve PDF katalog çalışır.

## 3. Kurulum

1. **Python'u kur** ([python.org](https://www.python.org/downloads/)). Kurulumda *"Add python.exe
   to PATH"* kutusunu işaretle.
2. **Projeyi indir.** Git ile:

   ```powershell
   git clone https://github.com/emrekaandrms/catalog-organizer.git
   cd catalog-organizer
   ```

   (Git yoksa GitHub sayfasındaki **Code → Download ZIP** ile indirip klasöre çıkar.)
3. **Sanal ortam oluştur ve etkinleştir:**

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

4. **Programı kur** (bağımlılıklar dahil; birkaç yüz MB iner, birkaç dakika sürer):

   ```powershell
   pip install -e .
   ```

5. **Başlat:**

   ```powershell
   python -m catalog_organizer.app
   ```

Her seferinde önce `.venv\Scripts\activate` yapmayı unutma. Komutu proje klasöründen çalıştır:
program ayar ve veri dosyalarını proje klasöründe arar.

> **Rhino kullanıyorsan** (isteğe bağlı): `pip install -e ".[rhino]"`. Rhino yoksa program belgelenmiş
> bir sezgisel yöntemle devam eder.

## 4. İlk açılış ve model seçimi

Açılışta program kısa bir **kontrol** yapar: seçili sağlayıcı Ollama ise sunucuya ve modele bakar,
diskte yer olup olmadığını denetler. Sorun varsa pencerede söyler.

### 4.1 Yerel model (önerilen): Ollama

1. [ollama.com](https://ollama.com) adresinden kur.
2. Varsayılan modeli indir:

   ```powershell
   ollama pull qwen3.5:9b-q4_K_M
   ```

3. Ollama arka planda çalışırken programı aç. Başka bir model kullanacaksan **Settings → VLM
   Provider** sekmesinden değiştir.

Görüntüler bilgisayarından çıkmaz.

### 4.2 Bulut sağlayıcı (isteğe bağlı)

**Settings → VLM Provider** sekmesinde sağlayıcıyı seç (OpenAI, MiniMax, GLM, OpenCode Zen veya
OpenAI uyumlu bir uç nokta), modeli ve adresi gir, **API anahtarını** yaz. Anahtar işletim
sisteminin kimlik bilgisi deposuna (Windows Kimlik Bilgisi Yöneticisi) kaydedilir; **hiçbir dosyaya
yazılmaz**. Bu durumda görüntüler seçtiğin sağlayıcıya gönderilir ve sağlayıcı ücret kesebilir;
program oturum maliyetini durum çubuğunda gösterir.

## 5. Arayüze genel bakış

Sol kenar çubuğunda dokuz sekme var. `Ctrl+1` … `Ctrl+9` ile geçiş yapılır.

| Sekme | Ne için |
|---|---|
| **Dashboard** | Özet sayılar, katalog dağılımı, son işlemler |
| **Analyze** | Yapay zekâsız geometri analizi: ağırlık, taş, ölçü, koçan. Tek dosya veya klasör |
| **Process** | Bir klasörde tam işlem hattı (görüntü + model + ölçüm) ve katalog CSV'si |
| **Search** | Katalogda filtre ve serbest metin araması |
| **Publish** | Seçim listeleri, ilan metni, fiyat, Etsy/WooCommerce CSV, PDF katalog |
| **Render** | Canlı 3D oynatıcı; maden ve taş rengi; açı kaydetme; PNG |
| **Diagnostics** | Tek bir dosyada her aşamayı ayrı çalıştır, sürelerini gör |
| **Settings** | Yollar, işlem hattı, model sağlayıcı, fiyatlar, tema |
| **Logs** | Denetim kaydını canlı izle, önem düzeyine göre süz |

Alt çubukta model sağlayıcısı, oturum maliyeti, toplu işlem durumu ve ürün sayısı görünür.

## 6. İş akışları

### 6.1 Hızlı geometri analizi (Analyze)

1. **Analyze** sekmesine geç.
2. **Analyze file…** ile bir dosya, **Analyze folder…** ile bir klasör seç.
3. Tablo dolar: boyutlar, ağırlıklar (sekiz alaşım), taşlar, koçan.
4. **Export CSV…** ile tabloyu kaydet.

Komut satırından aynısı: `python -m catalog_organizer.app analyze --folder KLASOR --csv cikti.csv`.

### 6.2 Arşivi kataloglama (Process)

1. **Process** sekmesine geç.
2. **Pick folder…** ile arşiv klasörünü seç. Program `.3dm` ve `.stl` dosyalarını bulur, her
   birine kalıcı bir kimlik (`JCAD-000000123` biçimi) verir.
3. **Start**'a bas. Her dosya için sırayla: yükleme → görüntü üretimi → model çağrısı →
   ölçüm → kayıt.
4. **Pause** / **Cancel** ile durdur. Kaldığı yerden devam eder; zaten işlenmiş dosyaları atlar.
5. **Open CSV** ile o ana kadarki sonucu aç (iş sürerken de açılabilir).

İlk çalıştırmada **az sayıda dosya** (5–20) ile dene, sonuçlara bak, sonra tüm arşive geç.
Modelin güveni düşük olan kayıtlar "incelenecek" diye işaretlenir (Dashboard'da sayısı görünür).

### 6.3 Arama ve seçim (Search → Publish)

1. **Search** sekmesinde kategori, alt kategori, taş durumu, marka, koçan filtrelerini ve serbest
   metni kullan; **Search**'e bas.
2. Sonuçlardan **Tick all / Untick all** ile işaretle, **Add checked** (işaretliler) veya
   **Add all matching** (tüm eşleşenler) ile aktif listeye ekle.
3. **Publish** sekmesinde **New list** ile liste oluştur; listeler orada saklanır.
4. **Correct category / brand** ile modelin yanlış bulduğu kategori veya markayı düzelt.

### 6.4 İlan metni, fiyat, CSV (Publish)

1. Bir liste seç. İşaretli ürünler varsa işlemler yalnızca onlara, yoksa tüm listeye uygulanır.
2. **Generate listing copy**: bir metin modeli her ürün için başlık, açıklama, etiket yazar.
   (Görüntü göndermez; analizde üretilen açıklamayı kullanır.) İş ayrı bir iş parçacığında
   yürür, pencere donmaz.
3. **Recalculate prices**: fiyatlar `config/pricing.yaml` ve **Settings → Pricing** değerlerinden
   hesaplanır. Fiyat formülleri `catalog_organizer/listing/pricing.py` içindedir. Her fiyat, hangi
   değerlerle hesaplandığının damgasını taşır; kur değişince hangi ilanların bayatladığı
   sorgulanabilir.
4. **Channel** olarak Etsy veya WooCommerce'i seç, **Export** ile CSV'yi yaz. WooCommerce
   CSV'si UTF-8-BOM yazılır (Excel'de Türkçe karakterler bozulmaz); SKU'lar kalıcıdır, yeniden
   içe aktarma mevcut ürünü günceller, kopya açmaz.

> **Önemli:** `config/pricing.yaml` içindeki sayılar **örnektir**. Kendi maliyet ve kur
> değerlerini gir; metal kurları aylar içinde eskir.

## 7. Render sekmesi ve PDF katalog

Render sekmesi, kataloğun **basacağı görüntünün aynısını** canlı gösterir: PDF katalog ve "PNG
kaydet" aynı motoru kullanır.

### 7.1 Kullanım

1. Soldan bir ürün seç (veya **Dosyadan aç…** ile katalogda olmayan bir `.3dm` / `.stl` dosyası).
2. İlk açılışta ürün hazırlanır: bu **saniyeler** sürer (STL'de taş yuvaları taranıyorsa
   **dakikalar**); sonra sonuç saklanır ve ürün **anında** açılır.
3. İki canlı oynatıcı yan yana: **KARŞIDAN** ve **ÇAPRAZ**, yani katalog sayfasının iki görünümü.
   Fareyle döndür, tekerlekle yakınlaştır.
4. **Maden** ve **Taş** listelerinden renk seç; değişiklik anında uygulanır.
5. Otomatik açı beğenilmediyse, parçayı döndür ve o panelin **Bu açıyı kaydet** düğmesine bas. O
   ürün için o görünüm artık bu açıdır; PDF ve PNG'ler de bunu kullanır. Geri dönmek için
   **Otomatiğe sıfırla**.
6. **PNG boyutu** listesinden boyutu seçip **PNG kaydet…** ile iki görünümü klasöre yaz.
7. **Yeniden çiz**: dosya değiştiyse sahneyi baştan hazırlar.

Aynı kategorideki tüm ürünler **aynı açıdan** çekilir (yüzükler baş gösterecek şekilde, düz
parçalar kameraya dönük); katalogda tutarlı görünür. Zemin düz açık gridir, gölge ve yansıma
yoktur: ürün perakende görüntüleyicilerindeki gibi gösterilir.

### 7.2 PDF katalog

1. **Publish** sekmesinde bir liste seç.
2. **PDF katalog** düğmesine bas; **maden** ve **taş** rengini sor.
3. Program her ürünü çizer ve A4 PDF yazar: kapak sayfası (liste adı, tarih, renkler), ardından
   **ürün başına bir sayfa**: karşıdan ve çapraz yan yana, altında ID, kategori, ağırlık, ölçü.

Gösterilen ağırlık, çizilen madenin ağırlığıdır (gümüşte gümüş gramı, altınlarda 14 ayar
gramı). Kaynak dosyası bulunamayan veya çizilemeyen ürün de sayfa alır; "görsel yok" notuyla.

## 8. Ayarlar ve yapılandırma dosyaları

**Settings** sekmesi şu YAML dosyalarını düzenler (`config/` klasörü):

| Dosya | İçerik |
|---|---|
| `app_settings.yaml` | Çıktı/önbellek/veri yolları, tema, kısayollar |
| `pipeline_settings.yaml` | Disk uyarısı, ölçek eşikleri, görüntü çözünürlüğü, **model sağlayıcı ayarları** |
| `categories.yaml` | Kategori ve alt kategoriler |
| `tag_dictionary.yaml` | Kontrollü etiket sözlüğü (modelin seçebileceği etiketler) |
| `vlm_prompt_templates.yaml` | Modelin görüntü sınıflandırma talimatı |
| `listing_prompt_templates.yaml` | İlan metni talimatları |
| `pricing.yaml` | Fiyat sabitleri (örnek değerler) |
| `metal_density_table.yaml` | Alaşım yoğunlukları |
| `stone_weight_tables/` | Taş boyutu → karat/gram tabloları (CZ) |
| `etsy_taxonomy_map.yaml` | Kategori → Etsy sınıflandırma eşlemesi |
| `brand_names.yaml` | Modelin tanıyabileceği marka listesi |

Çoğunu Ayarlar ekranından değiştirebilirsin; elle düzenlemek de serbest. API anahtarları bu
dosyalarda **bulunmaz**.

## 9. STL mi, 3DM mi?

İki dosya türü aynı parçayı çok farklı anlatır:

| | `.3dm` (Rhino / MatrixGold) | `.stl` |
|---|---|---|
| Taşlar | Ayrı bir **katmanda** gerçek geometri olarak durur | **Yoktur.** STL, makineye girmeden önceki haldir |
| Taş bilgisi | Katman adından ve geometriden **birebir** okunur | Yalnızca taşların oturacağı **yuvalar** vardır; program ölçer |
| Boyut/ağırlık | Katmanlardan, kesicilerden hassas | Tek gövde; daha az bilgi |
| Güvenilirlik | Yüksek | Tahmin |

**Öneri:** Taşın yeri, boyu ve adedi önemliyse **`.3dm`** ver. STL vermen sorun değil, ama taş
bilgisi bir tahmin olur.

### 9.1 `.3dm` dosyası nasıl hazırlanmalı

* Taşları adında `gem` veya `stone` geçen bir **katmana** koy; program taş geometrisini oradan okur.
* Taş yuvası kesicilerini `Cutter` / `Cutting Objects` benzeri bir katmanda tut; ağırlıktan
  düşülür.
* Parçanın yanına not, müşteri adı gibi yazılar koyma: program bunları ayıklamaya çalışır ama
  yanlış sayabilir.

(Katman adı kuralları: `src/catalog_organizer/cad/stone_extraction.py`.)

### 9.2 STL'de "Taşları yuvalara yerleştir"

STL'de taş olmadığı için görüntüde taş görünmez. Program, taş yuvalarını bulup oraya standart bir
briyan koyabilir. Bu bir **tahmindir** ve şu kurallarla çalışır:

* Varsayılan **kapalıdır**. Ürünün katalog kaydı **"taşlı"** diyorsa açık gelir; **"taşsız"**
  (polished) diyorsa **hiçbir zaman** taş konmaz.
* Render sekmesindeki **Taşları yuvalara yerleştir** kutusu **senin kararındır**: ürün başına
  açıp kapatırsın, seçim kalıcıdır ve **PDF katalog ile PNG de aynı kararı** kullanır.
* Kutu yalnızca STL'de etkindir; `.3dm`'de taşlar dosyadan gelir.
* Program yalnızca **yuvarlak, koni gibi daralan, üstü açık** yuvaları sayar. Pavé kanalları,
  delikli bantlar, içeriden açılmış galeri delikleri taş almaz. Çok eğik duran yuvalar
  görünmeyebilir.
* Görsel olarak mutlaka kontrol et. Yanlış yere taş konduysa kutuyu kapat.
* Büyük bir STL'nin ilk açılışında yuva taraması **bir-iki dakika** sürebilir; sonuç saklanır.

Parçan taşsızsa (gerçekten düz bir yüzük) ve kaydı yanlışlıkla "taşlı" ise, kutuyu kapatman yeterli.

## 10. Verilerin yeri, yedekleme, sıfırlama

Hepsi **proje klasörünün** altındadır (yolları `app_settings.yaml` ile değiştirebilirsin):

| Klasör | İçerik | Silinirse |
|---|---|---|
| `data/` | `catalog.db` (katalog, listeler, seçimler, kamera ve render tercihleri), `catalog_master.jsonl`, denetim kaydı | **Katalog gider.** Yedekle. |
| `cache/` | Görüntüler, render sonuçları, 3D sahneler | Güvenle silinebilir; yeniden üretilir |

**Yedek:** programı kapatıp `data/` klasörünü kopyala.
**Önbelleği temizle:** programı kapatıp `cache/` klasörünü sil (ilk açılışlar yeniden yavaş olur).

## 11. Sorun giderme

**"3D oynatıcı için PyQt6-WebEngine gerekli"**
`pip install -e .` ile yeniden kur ya da `pip install "PyQt6-WebEngine>=6.7,<6.8"`.

**Render sekmesi boş / "görüntüleyici hazır olmadı"**
Ekran kartı sürücünü güncelle. WebGL 2 gerekir; uzak masaüstü ve bazı sanal makinelerde yoktur.
Tarayıcında `webglreport.com` adresiyle WebGL 2'yi denetleyebilirsin.

**Render çok yavaş açılıyor**
Yalnızca bir ürünün **ilk** açılışında. Sonuç `cache/` içinde saklanır. Milyonlarca üçgenli
dosyalar 20 saniyeye, taşsız STL'de yuva taraması birkaç dakikaya çıkabilir.

**"Ollama'ya ulaşılamıyor" / sınıflandırma çalışmıyor**
Ollama'nın çalıştığını (`ollama list`) ve modelin indirildiğini kontrol et. Settings → VLM
Provider'daki adres ve port doğru olmalı (varsayılan `127.0.0.1:11434`).

**Taşlar yanlış yere / taşsız üründe taş görünüyor**
Bölüm 9.2: kutuyu kapat. Katalog kaydı yanlışlıkla "taşlı" ise Publish → Correct bölümünden veya
yeniden analizle düzelt.

**Ağırlık terazide çıkandan farklı**
Bölüm 12. Ağırlıklar tahmindir; özellikle üst üste binen gövdeler, ince cidarlar ve STL
dosyalarında sapma olur.

**Program hiç açılmıyor**
Komutu proje klasöründen ve sanal ortam etkinken çalıştır. Hata mesajını bir *issue* açarken
paylaş; **müşteri dosyalarını paylaşma**.

## 12. Doğruluk ve sınırlar

* **Ağırlıklar** az sayıda gerçek dosyada atölye terazisine göre ayarlandı; tipik olarak birkaç
  yüzde, bazı dosyalarda daha fazla sapar. **Fiyatlamadan önce terazi ile doğrula.**
* **Taş bilgisi:** `.3dm`'de geometriden okunur; STL'de yuvalardan tahmin edilir; model yalnızca
  görünen taş sayısını tahmin eder.
* **Sınıflandırma** seçtiğin modelin kalitesine bağlıdır; güveni düşük kayıtlar incelemeye
  işaretlenir. Marka tanıma modelin genel bilgisine dayanır, düşük güvenli bir yoldur.
* **Render** WebGL 2 ister; ilk açılışlar yavaştır.
* Yalnızca **Windows 11**'de test edildi.

Hata bildirmek ve katkı vermek için: [CONTRIBUTING.md](../CONTRIBUTING.md).
