' ЛР-4. Моделі з трендом, АРІКС, АР, АРКС і статичні/динамічні прогнози (бригада 15, варіант 1)
' Аргумент: %0 - папка з даними (без пробілів, із "\" в кінці)
' Приклад запуску: run lab4 d:\ats_lab_04_new\data\
' Навчальна вибірка - до 2007:12, перевірочна - 2008:01..2008:08 (8 значень)

' =============================== RSAFSNA ===============================
wfcreate(wf=rsafsna) m 1992:01 2008:08
read(t=txt) {%0}rsafsna.txt y
series k = @trend + 1
series logy = log(y)
series dlogy = d(logy)
freeze(gr_y) y.line
freeze(gr_logy) logy.line

smpl 1992:01 2007:12
freeze(tab_corr_logy) logy.correl(24)
freeze(tab_corr_dlogy) dlogy.correl(36)

' ---------- Тренди 1-3 порядку ----------
equation trend1.ls logy c k
equation trend2.ls logy c k k^2
equation trend3.ls logy c k k^2 k^3

' ---------- АРІКС: модель для перших різниць ----------
equation ar_d.ls dlogy c ar(1) ar(2) ar(3) ar(4) ar(12)
equation arima.ls dlogy c ar(1) ar(4) ar(12) ma(1) ma(3) ma(4) ma(9) ma(10)

' ---------- АР та АРКС для logy ----------
equation ar.ls logy c logy(-1) logy(-12) logy(-13)
equation arma.ls logy c ar(1) ar(12) ar(13) ma(3) ma(6) ma(9)

' ---------- Прогнози: fit - статичний, forecast - динамічний ----------
smpl 2008:01 2008:08
trend1.fit(e) logy_t1
trend2.fit(e) logy_t2
trend3.fit(e) logy_t3
arima.fit(e) dlogy_fs
arima.forecast(e) dlogy_fd
ar.fit(e) logy_ar_s
ar.forecast(e) logy_ar_d
arma.fit(e) logy_arma_s
arma.forecast(e) logy_arma_d

' АРІКС -> logy: статичний logy(k) = dlogy_f(k) + logy(k-1), динамічний - через попередній прогноз
series logy_arima_s = dlogy_fs + logy(-1)
series logy_arima_d = na
smpl 2008:01 2008:01
logy_arima_d = dlogy_fd + logy(-1)
smpl 2008:02 2008:08
logy_arima_d = dlogy_fd + logy_arima_d(-1)
smpl 2008:01 2008:08

group g_static logy logy_t1 logy_t2 logy_t3 logy_arima_s logy_ar_s logy_arma_s
group g_dynamic logy logy_arima_d logy_ar_d logy_arma_d
freeze(gr_static) g_static.line
freeze(gr_dynamic) g_dynamic.line

' ---------- Показники якості прогнозів ----------
table(11,5) tab_fc
tab_fc(1,1) = "Forecast"
tab_fc(1,2) = "RMSE"
tab_fc(1,3) = "MAE"
tab_fc(1,4) = "MAPE"
tab_fc(1,5) = "Theil U"
!r = 2
for %f logy_t1 logy_t2 logy_t3 logy_arima_s logy_ar_s logy_arma_s logy_arima_d logy_ar_d logy_arma_d
  tab_fc(!r,1) = %f
  tab_fc(!r,2) = @sqrt(@mean((logy - {%f})^2))
  tab_fc(!r,3) = @mean(@abs(logy - {%f}))
  tab_fc(!r,4) = 100*@mean(@abs((logy - {%f})/logy))
  tab_fc(!r,5) = @sqrt(@mean((logy - {%f})^2))/(@sqrt(@mean({%f}^2)) + @sqrt(@mean(logy^2)))
  !r = !r + 1
next
show tab_fc
smpl @all

' =============================== CURRNS ===============================
wfcreate(wf=currns) m 1947:01 2008:08
read(t=txt) {%0}currns.txt y
series k = @trend + 1
series logy = log(y)
series dlogy = d(logy)
freeze(gr_y) y.line
freeze(gr_logy) logy.line

smpl 1947:01 2007:12
freeze(tab_corr_logy) logy.correl(24)
freeze(tab_corr_dlogy) dlogy.correl(36)

equation trend1.ls logy c k
equation trend2.ls logy c k k^2
equation trend3.ls logy c k k^2 k^3

equation ar_d.ls dlogy c ar(5) ar(10) ar(12)
equation arima.ls dlogy c ar(12) ma(1) ma(3) ma(6) ma(9) ma(12)

equation ar.ls logy c logy(-1)
equation arma.ls logy c ar(1) ma(1) ma(2) ma(5) ma(7) ma(10) ma(12)

smpl 2008:01 2008:08
trend1.fit(e) logy_t1
trend2.fit(e) logy_t2
trend3.fit(e) logy_t3
arima.fit(e) dlogy_fs
arima.forecast(e) dlogy_fd
ar.fit(e) logy_ar_s
ar.forecast(e) logy_ar_d
arma.fit(e) logy_arma_s
arma.forecast(e) logy_arma_d

series logy_arima_s = dlogy_fs + logy(-1)
series logy_arima_d = na
smpl 2008:01 2008:01
logy_arima_d = dlogy_fd + logy(-1)
smpl 2008:02 2008:08
logy_arima_d = dlogy_fd + logy_arima_d(-1)
smpl 2008:01 2008:08

group g_static logy logy_t1 logy_t2 logy_t3 logy_arima_s logy_ar_s logy_arma_s
group g_dynamic logy logy_arima_d logy_ar_d logy_arma_d
freeze(gr_static) g_static.line
freeze(gr_dynamic) g_dynamic.line

table(11,5) tab_fc
tab_fc(1,1) = "Forecast"
tab_fc(1,2) = "RMSE"
tab_fc(1,3) = "MAE"
tab_fc(1,4) = "MAPE"
tab_fc(1,5) = "Theil U"
!r = 2
for %f logy_t1 logy_t2 logy_t3 logy_arima_s logy_ar_s logy_arma_s logy_arima_d logy_ar_d logy_arma_d
  tab_fc(!r,1) = %f
  tab_fc(!r,2) = @sqrt(@mean((logy - {%f})^2))
  tab_fc(!r,3) = @mean(@abs(logy - {%f}))
  tab_fc(!r,4) = 100*@mean(@abs((logy - {%f})/logy))
  tab_fc(!r,5) = @sqrt(@mean((logy - {%f})^2))/(@sqrt(@mean({%f}^2)) + @sqrt(@mean(logy^2)))
  !r = !r + 1
next
show tab_fc
smpl @all
