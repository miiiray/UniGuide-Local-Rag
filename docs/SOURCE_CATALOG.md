# Resmî Kaynak Kataloğu

Son erişim tarihi: **26 Ağustos 2026**

Bu proje yalnızca Haliç Üniversitesi'nin resmî alan adındaki belgeleri hedefler.
Belgeler değişebileceği için teslim ve demo öncesinde güncel sürümler yeniden
indirilmelidir.

## Ana kaynaklar

1. [Haliç Üniversitesi Ön Lisans ve Lisans Eğitim-Öğretim Yönergesi (PDF)](https://halic.edu.tr/wp-content/uploads/universitemiz/mevzuat/yonergeler/halic-universitesi-on-lisans-ve-lisans-yonergesi.pdf)
2. [Haliç Üniversitesi Lisans ve Ön Lisans Uygulamalı Dersler Staj Yönergesi (PDF)](https://halic.edu.tr/tr/s-universitemiz/Documents/mevzuat/yonergeler/halic-universitesi-lisans-ve-on-lisans-uygulamali-dersler-staj-yonergesi.pdf)
3. [Haliç Üniversitesi Yatay Geçiş sayfası](https://halic.edu.tr/tr/ogrencimiz/yatay-gecis)
4. [Haliç Üniversitesi Uygulamalı Eğitimler ve Staj Esasları](https://halic.edu.tr/tr/ogrencimiz/uygulamali-egitimler-ve-staj-koordinatorlugu/uygulamali-egitimler-ve-staj-esaslari)
5. [Bilgisayar Mühendisliği (İngilizce) program sayfası](https://halic.edu.tr/tr/akademik/fakulteler/muhendislik-fakultesi/bolumler/bilgisayar-muhendisligi-ingilizce)
6. [Haliç Üniversitesi Mevzuat / Yönergeler dizini](https://halic.edu.tr/tr/universitemiz/mevzuat/yonergeler)
7. [2026-2027 Güz Dönemi Kurum İçi Yatay Geçiş ve ÇAP-Yandal Duyurusu](https://halic.edu.tr/tr/duyurular/2026-2027-akademik-yili-guz-donemi-basariya-dayali-kurum-ici-yatay-gecis-ve-cift-ana-dal-yan-dal-basvuru-islemleri)
8. [2026-2027 Güz Dönemi Kurumlar Arası Yatay Geçiş Duyurusu](https://halic.edu.tr/tr/duyurular/2026-2027-egitim-ogretim-yili-guz-yariyili-kurumlararasi-genel-not-ortalamasi-merkezi-yerlestirme-puani-ek-madde-1-ve-yurt-disi-yatay-gecis-basvurulari)

## PDF ekleme

Yukarıdaki PDF'leri tarayıcıdan indirip `data/official/` klasörüne yerleştirin. Sonra:

```powershell
uniguide index --rebuild
```

Resmî PDF'ler Git deposuna eklenmez. Böylece eski bir mevzuat kopyasının yanlışlıkla
kalıcı kaynak olarak dağıtılması önlenir.

Web duyurularını veri kümesine eklemek için ilgili sayfayı tarayıcıda açın, `Ctrl+P`
ile **PDF olarak kaydet** seçeneğini kullanın ve çıktıyı `data/official/` klasörüne
yerleştirin. `official/` içinde belge bulunduğunda demo özetleri otomatik olarak
dışarıda bırakılır.
