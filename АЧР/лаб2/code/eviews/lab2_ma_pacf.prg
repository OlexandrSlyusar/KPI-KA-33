' ЛР-2. Аналіз часових рядів. КПІ ім. Ігоря Сікорського, ННІ ПСА, 2026
' %0 - ім'я робочого файлу (*.wf1), наприклад rts1
' %1 - ім'я ряду в робочому файлі, наприклад rts1
' %2 - розмір вікна ковзного середнього N (5 або 10)
' %3 - кількість лагів для АКФ/ЧАКФ (12)

' 2.3.1 Завантаження робочого файлу, ім'я передається параметром
load {%0}
smpl @all
!n = {%2}
!lags = {%3}
!obs = @obs({%1})

' 2.3.2 Ваги експоненційного КС: alpha = 2/(N+1)
!alpha = 2/(!n+1)
vector(!n) w{%2}
!sw = 0
for !i = 1 to !n
w{%2}(!i) = (1-!alpha)^(!n-!i+1)
!sw = !sw + w{%2}(!i)
next

' Ряди простого (_SMA) та експоненційного (EMA) КС
series _sma{%2}
series ema{%2}
for !k = !n to !obs
!s1 = 0
!s2 = 0
for !i = 1 to !n
!s1 = !s1 + {%1}(!k-!n+!i)
!s2 = !s2 + w{%2}(!i)*{%1}(!k-!n+!i)
next
_sma{%2}(!k) = !s1/!n
ema{%2}(!k) = !s2/!sw
next

' 2.3.3 Вибіркова АКФ r(s) (формула 13.2)
!mean = @mean({%1})
!den = @sum(({%1}-!mean)^2)
vector(!lags) r
for !s = 1 to !lags
!num = 0
for !k = !s+1 to !obs
!num = !num + ({%1}(!k)-!mean)*({%1}(!k-!s)-!mean)
next
r(!s) = !num/!den
next

' ЧАКФ за рекурентною формулою (13.1), вкладені цикли
matrix(!lags,!lags) phi
vector(!lags) chakf
phi(1,1) = r(1)
chakf(1) = r(1)
for !k = 2 to !lags
!num = r(!k)
!dd = 1
for !j = 1 to !k-1
!num = !num - phi(!k-1,!j)*r(!k-!j)
!dd = !dd - phi(!k-1,!j)*r(!j)
next
phi(!k,!k) = !num/!dd
for !j = 1 to !k-1
phi(!k,!j) = phi(!k-1,!j) - phi(!k,!k)*phi(!k-1,!k-!j)
next
chakf(!k) = phi(!k,!k)
next
show chakf

' Перевірка: стандартна корелограма EViews
freeze(tab_correl) {%1}.correl({%3})
show tab_correl

' 2.3.4 Графіки: на одній площині індекс, _SMA та EMA
group g_all{%2} {%1} _sma{%2} ema{%2}
freeze(gr_all{%2}) g_all{%2}.line
show gr_all{%2}

' Графіки для звіту - перші 50 точок
smpl 1 50
group g_sma{%2} {%1} _sma{%2}
group g_ema{%2} {%1} ema{%2}
freeze(gr_sma{%2}) g_sma{%2}.line
freeze(gr_ema{%2}) g_ema{%2}.line
show gr_sma{%2}
show gr_ema{%2}
smpl @all
