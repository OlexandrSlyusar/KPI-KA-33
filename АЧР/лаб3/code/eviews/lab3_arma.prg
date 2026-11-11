' ЛР-3. Частина 2: описова статистика та підбір рівняння АРКС(p,q)
' Аргументи: %0 - папка з даними (без пробілів, із "\" в кінці), %1 - файл, %2 - ім'я ряду,
'           %3 - розмір вибірки, %4 - p, %5 - q (за ЧАКФ залишків АР(p)),
'           %6..%9 - q для КС по y: просте N=5, просте N=10, експоненційне N=5, експоненційне N=10
' Бригада 15 (варіант 1):
'   run lab3_arma d:\ats_lab_03_new\data\ rts1.txt rts1 178 2 5 1 1 1 1
'   run lab3_arma d:\ats_lab_03_new\data\ 1996rts1.txt rts1996 249 3 7 2 1 2 2

wfcreate(wf={%2}) u {%3}
read(t=txt) {%0}{%1} {%2}
series y = {%2}
!p = {%4}
!q = {%5}
%qy = %6 + " " + %7 + " " + %8 + " " + %9

' ---------- Описова статистика (View -> Descriptive Statistics -> Histogram and Stats) ----------
freeze(hist_y) y.hist
show hist_y

' ---------- 1.1 АКФ та ЧАКФ ряду для 12 лагів -> p ----------
freeze(tab_corr_y) y.correl(12)

%ylags = ""
%arterms = ""
for !i = 1 to !p
  %ylags = %ylags + " y(-" + @str(!i) + ")"
  %arterms = %arterms + " ar(" + @str(!i) + ")"
next
%materms = ""
for !j = 1 to !q
  %materms = %materms + " ma(" + @str(!j) + ")"
next

' ---------- 1.2 АР(p) та ЧАКФ його залишків -> q ----------
equation eq_ar.ls y c {%ylags}
series r1 = resid
freeze(tab_corr_r1) r1.correl(12)

' ---------- Підхід 1: КС системи EViews ----------
equation eq_arma_ev.ls y c {%arterms} {%materms}

' ---------- Підхід 2: власні КС по залишках АР(p) ----------
series mv_s5 = @movav(r1, 5)
series mv_s10 = @movav(r1, 10)
series mvy_s5 = @movav(y, 5)
series mvy_s10 = @movav(y, 10)

' експоненційне КС (програма ЛР-2): w(i) = (1-alpha)^(N-i+1), alpha = 2/(N+1)
for !n = 5 to 10 step 5
  !alpha = 2/(!n+1)
  !sw = 0
  series mv_e{!n} = 0
  series mvy_e{!n} = 0
  for !i = 1 to !n
    !w = (1-!alpha)^(!n-!i+1)
    !lg = !i-!n
    mv_e{!n} = mv_e{!n} + !w*r1(!lg)
    mvy_e{!n} = mvy_e{!n} + !w*y(!lg)
    !sw = !sw + !w
  next
  mv_e{!n} = mv_e{!n}/!sw
  mvy_e{!n} = mvy_e{!n}/!sw
next

for %m mv_s5 mv_s10 mv_e5 mv_e10
  %mvl = ""
  for !j = 1 to !q
    %mvl = %mvl + " " + %m + "(-" + @str(!j) + ")"
  next
  series y1_{%m} = y - {%m}
  equation eq_r_{%m}.ls y1_{%m} c {%ylags} {%mvl}
next

' ---------- 2. КС по вихідному сигналу y ----------
!k = 0
for %m mvy_s5 mvy_s10 mvy_e5 mvy_e10
  !k = !k + 1
  !qm = @val(@word(%qy, !k))
  freeze(tab_corr_{%m}) {%m}.correl(12)
  %mvl = ""
  for !j = 1 to !qm
    %mvl = %mvl + " " + %m + "(-" + @str(!j) + ")"
  next

  ' Підхід 1: власні коефіцієнти b(i) = (1-alpha)^i / sum (1-alpha)^j, alpha = 2/(q+1)
  vector(!qm) b_{%m}
  if !qm = 1 then
    b_{%m}(1) = 1
  else
    !alpha = 2/(!qm+1)
    !s = 0
    for !j = 1 to !qm
      !s = !s + (1-!alpha)^!j
    next
    for !j = 1 to !qm
      b_{%m}(!j) = (1-!alpha)^!j/!s
    next
  endif
  series y1_{%m} = y - {%m}
  for !j = 1 to !qm
    !c = b_{%m}(!j)
    !lg = -!j
    y1_{%m} = y1_{%m} - !c*{%m}(!lg)
  next
  equation eq_y1_{%m}.ls y1_{%m} c {%ylags}

  ' Підхід 2: коефіцієнти при КС оцінюються разом з a0..ap
  series y2_{%m} = y - {%m}
  equation eq_y2_{%m}.ls y2_{%m} c {%ylags} {%mvl}
next

' ---------- Таблиця 2: R-squared, Sum squared resid, Akaike, Durbin-Watson ----------
table(15,6) tab2
tab2(1,1) = "Equation"
tab2(1,2) = "R-squared"
tab2(1,3) = "Sum squared resid"
tab2(1,4) = "Akaike"
tab2(1,5) = "Durbin-Watson"
tab2(1,6) = "Schwarz"
!r = 2
for %e eq_ar eq_arma_ev eq_r_mv_s5 eq_r_mv_s10 eq_r_mv_e5 eq_r_mv_e10 eq_y1_mvy_s5 eq_y1_mvy_s10 eq_y1_mvy_e5 eq_y1_mvy_e10 eq_y2_mvy_s5 eq_y2_mvy_s10 eq_y2_mvy_e5 eq_y2_mvy_e10
  tab2(!r,1) = %e
  tab2(!r,2) = {%e}.@r2
  tab2(!r,3) = {%e}.@ssr
  tab2(!r,4) = {%e}.@aic
  tab2(!r,5) = {%e}.@dw
  tab2(!r,6) = {%e}.@schwarz
  !r = !r + 1
next
show tab2
