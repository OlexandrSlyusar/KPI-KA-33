' ЛР-3. Частина 1: обчислення статистики Дарбіна-Уотсона
' Аргумент програми: %0 - ім'я ряду (вектор часових даних, наприклад залишки моделі)
' Приклад запуску: run lab3_dw e   (Program arguments: e)
' Перевірка: ряд з файлу example_for_DW.txt (workfile u 1 20) -> DW = 1.22881

smpl @all
smpl if {%0} <> na
stom({%0}, _v)
!n = @rows(_v)

' DW = sum_{t=2..N}(e(t)-e(t-1))^2 / sum_{t=1..N} e(t)^2
!num = 0
!den = _v(1)^2
for !t = 2 to !n
  !num = !num + (_v(!t) - _v(!t-1))^2
  !den = !den + _v(!t)^2
next
scalar dw_num = !num
scalar dw_den = !den
scalar dw = !num/!den
smpl @all

table(4,2) tab_dw
tab_dw(1,1) = "Series"
tab_dw(1,2) = %0
tab_dw(2,1) = "Observations"
tab_dw(2,2) = !n
tab_dw(3,1) = "Sum (e(t)-e(t-1))^2"
tab_dw(3,2) = dw_num
tab_dw(4,1) = "Durbin-Watson stat"
tab_dw(4,2) = dw
show tab_dw
delete _v
