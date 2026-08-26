# UniGuide Local RAG

UniGuide, Haliç Üniversitesi öğrenci belgeleri üzerinde çalışan, yanıtlarını kaynak
metinlere dayandıran ve Microsoft Foundry Local sayesinde model çıkarımını cihazda
yapan bir RAG (Retrieval-Augmented Generation) uygulamasıdır.

> Bu proje eğitim ve demonstrasyon amaçlıdır. Üretilen yanıtlar resmî akademik
> danışmanlık yerine geçmez; kesin işlemler için üniversitenin güncel mevzuatı ve
> yetkili birimleri kontrol edilmelidir.

## Problem

Üniversite yönerge ve yönetmelikleri uzun, farklı sayfalara dağılmış ve zamanla
güncellenebilen belgelerdir. Genel amaçlı bir dil modeli bu kuralları ezberden yanlış
veya güncel olmayan biçimde aktarabilir. UniGuide önce yerel belgelerde ilgili bölümleri
bulur, sonra yalnızca bu bağlamı kullanarak cevap üretir.

## Özellikler

- PDF, Markdown ve düz metin belgelerini okuma
- Metni örtüşmeli chunk'lara ayırma
- Foundry Local ile cihaz üzerinde embedding üretme
- Embedding ve belge metinlerini SQLite'ta kalıcı saklama
- Cosine similarity ile Top-K anlamsal arama
- Kaynak dosya, PDF sayfası ve benzerlik skorunu gösterme
- Belge değişmediyse tekrar embedding üretmeme
- Belgede bilgi yoksa güvenli geri dönüş cevabı
- Terminal arayüzü ve Streamlit web arayüzü
- Bulut hesabı veya API anahtarı gerektirmeyen yerel model çıkarımı

## RAG akışı

1. `data/` altındaki belgeler okunur.
2. Her belge yaklaşık 1.200 karakterlik, 180 karakter örtüşmeli parçalara ayrılır.
3. `qwen3-embedding-0.6b` her parçayı sayısal bir vektöre dönüştürür.
4. Chunk, kaynak bilgisi ve embedding `storage/uniguide.db` dosyasına kaydedilir.
5. Kullanıcı sorusu aynı embedding modeliyle vektöre çevrilir.
6. Cosine similarity ile en yakın üç chunk seçilir.
7. Seçilen bağlam ve soru Türkçe destekleyen `phi-4-mini` sohbet modeline gönderilir.
8. Üretilen cevabın getirilen kaynakla örtüştüğü doğrulanır; örtüşmüyorsa doğrudan
   ilgili kaynak cümlesi güvenli cevap olarak kullanılır.
9. Yanıt ve kullanılan kaynaklar kullanıcıya gösterilir.

## Gereksinimler

- Windows 10/11
- Python 3.11 veya üzeri
- En az 8 GB RAM
- İlk model indirmesi için internet bağlantısı
- Modeller için yeterli boş disk alanı

İlk indirmeden sonra model çalışması ve belge araması yerel olarak gerçekleştirilebilir.

## Windows kurulumu

PowerShell'de:

```powershell
git clone https://github.com/miiiray/UniGuide-Local-RAG.git
cd UniGuide-Local-RAG

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

Henüz GitHub deposu oluşturulmadıysa aynı komutları yerel proje klasöründe `git clone`
satırı olmadan uygulayın.

## Belge ekleme

Demo belgeleri `data/demo/` klasöründedir. Güncel resmî PDF'leri
`data/official/` klasörüne ekleyin. Desteklenen uzantılar:

- `.pdf`
- `.md`
- `.txt`

Kullanılan resmî bağlantılar için [docs/SOURCE_CATALOG.md](docs/SOURCE_CATALOG.md)
dosyasına bakın. PDF dosyaları güncellenebildiği için depoya sabitlenmemiştir.

## Çalıştırma

### 1. Belgeleri indeksle

```powershell
uniguide index
```

Belgeler veya ayarlar değiştiyse sıfırdan indekslemek için:

```powershell
uniguide index --rebuild
```

### 2. Terminalden soru sor

```powershell
uniguide ask "Yandal programına en geç hangi yarıyılda başvurabilirim?"
uniguide chat
```

### 3. Web arayüzünü aç

```powershell
streamlit run src/uniguide/streamlit_app.py
```

Tarayıcıda genellikle `http://localhost:8501` adresi açılır. Önce sol menüden
**Belgeleri indeksle** düğmesine basın.

## Örnek sorular

- Yandal programına ne zaman başvurabilirim?
- Yandal için gereken minimum not ortalaması nedir?
- Yaz öğretiminde kaç AKTS ders alınabilir?
- Kurumlar arası yatay geçiş için not ortalaması kaç olmalıdır?
- Yatay geçiş başvurusunda hangi belgeler gereklidir?
- Bilgisayar Mühendisliği öğrencilerinin toplam staj süresi kaç iş günüdür?
- Staj yerini öğrenci kendisi bulabilir mi?
- Belgelerde yemekhane menüsü hakkında bilgi var mı?

Son soru, sistemin bağlam dışı sorularda bilgi uydurmadığını göstermek için kullanılır.

## Testler

Unit testler Foundry Local modelini indirmeden çalışır; sahte embedding runtime'ı ile
chunking, SQLite ve retrieval davranışını doğrular:

```powershell
python -m unittest discover -s tests -v
```

Manuel değerlendirme soruları ve başarı ölçütleri için
[docs/EVALUATION.md](docs/EVALUATION.md) dosyasına bakın.

## Proje yapısı

```text
UniGuide-Local-RAG/
├── data/
│   ├── demo/                 # Çevrimdışı çalışan küçük tanıtım veri seti
│   └── official/             # Kullanıcının ekleyeceği güncel resmî PDF'ler
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   └── SOURCE_CATALOG.md
├── src/uniguide/
│   ├── cli.py                # Terminal komutları
│   ├── config.py             # Ortam ayarları
│   ├── database.py           # SQLite veri katmanı
│   ├── documents.py          # Belge okuma ve chunking
│   ├── foundry.py            # Foundry Local entegrasyonu
│   ├── rag.py                # İndeks, retrieval ve generation akışı
│   └── streamlit_app.py      # Web arayüzü
└── tests/
```

## Tasarım kararları

- **Neden Foundry Local?** Belge metinleri ve model çıkarımı cihazdan çıkmaz; API
  anahtarı ve bulut hesabı gerekmez.
- **Neden SQLite?** Küçük bir üniversite belge koleksiyonu için kurulumsuz, taşınabilir
  ve anlaşılırdır.
- **Neden özel vector database yok?** Küçük veri kümesinde tüm vektörleri okuyup cosine
  similarity hesaplamak yeterlidir ve RAG mantığını görünür tutar.
- **Neden iki model?** Embedding modeli arama için; chat modeli doğal dilde cevap
  üretmek için ayrı görevler gerçekleştirir.
- **Neden kaynak ve skor gösteriliyor?** Kullanıcının cevabın hangi metne dayandığını
  denetleyebilmesi için.

## Sınırlamalar

- Taranmış ve metin katmanı olmayan PDF'ler OCR olmadan okunamaz.
- Küçük chat modellerinin Türkçe anlatım kalitesi daha büyük modellere göre sınırlıdır.
- Varsayılan `phi-4-mini`, ilk kullanımda birkaç GB model indirmesi gerektirir.
- Benzerlik eşiği tüm belge koleksiyonlarında yeniden değerlendirilmelidir.
- Mevzuat güncellendiğinde PDF yeniden indirilmeli ve indeks yeniden oluşturulmalıdır.
- Sistem hukuki veya resmî akademik karar vermez.

## Geliştiren

**Miray Şahin** — Bilgisayar Mühendisliği

## Kaynaklar

- [Microsoft Learn: Build a RAG application with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app)
- [Microsoft Learn: Foundry Local SDK reference](https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current)
- [Haliç Üniversitesi Mevzuat ve Yönergeler](https://halic.edu.tr/tr/universitemiz/mevzuat/yonergeler)
