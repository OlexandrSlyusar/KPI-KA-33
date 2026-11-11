' ЛР-3. Частина 4: часткова кореляційна функція ЧКФ(y, x) та модель ARMAX для rts1
' Запускається після lab3_mr.prg у тому ж робочому файлі. Аргумент: %0 - кількість лагів (12)
' ЧКФ(k) = corr( y(t), x(t-k) | x(t), x(t-1), ..., x(t-k+1) ), k = 0..K
' Обчислюється як кореляція залишків двох допоміжних регресій на c, x(t), ..., x(t-k+1)

!kmax = {%0}
table(!kmax+3, 8) tab_pccf
tab_pccf(1,1) = "Lag"
!col = 1
for %x rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl
  !col = !col + 1
  tab_pccf(1,!col) = %x
  vector(!kmax+1) pccf_{%x}
  pccf_{%x}(1) = @cor(rts1, {%x})
  for !k = 1 to !kmax
    %ctrl = %x
    for !j = 1 to !k-1
      %ctrl = %ctrl + " " + %x + "(-" + @str(!j) + ")"
    next
    smpl @first+!k @last
    equation _ey.ls rts1 c {%ctrl}
    series _ry = resid
    equation _ex.ls {%x}(-!k) c {%ctrl}
    series _rx = resid
    pccf_{%x}(!k+1) = @cor(_ry, _rx)
    smpl @all
  next
  for !k = 0 to !kmax
    tab_pccf(!k+2,1) = !k
    tab_pccf(!k+2,!col) = pccf_{%x}(!k+1)
  next
next
!band = 2/@sqrt(@obs(rts1)-!kmax)
tab_pccf(!kmax+3,1) = "+/-2/sqrt(T)"
tab_pccf(!kmax+3,2) = !band
show tab_pccf
delete _ey _ex _ry _rx

' ---------- Моделі ARMAX ----------
' 1) АР-частина rts1 (p = 2 за ЧАКФ) + регресори + лаги регресорів зі значущою ЧКФ(1)
equation eq_armax1.ls rts1 c rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl rts1(-1) rts1(-2) rtscr(-1) rtseu(-1) rtsfn(-1) rtstl(-1)
' 2) перші лаги всіх регресорів
equation eq_armax2.ls rts1 c rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl rts1(-1) rts1(-2) rtscr(-1) rtseu(-1) rtsfn(-1) rtsin(-1) rtsmm(-1) rtsog(-1) rtstl(-1)
' 3) після послідовного вилучення незначущих членів (Prob > 0.05)
equation eq_armax3.ls rts1 rtscr rtsfn rtsmm rtsog rtstl rts1(-1) rts1(-2) rtscr(-1) rtsfn(-1) rtsmm(-1) rtsog(-1) rtstl(-1)
' 4) регресія з АР(1)-похибкою засобами EViews
equation eq_armax4.ls rts1 c rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl ar(1)
show eq_armax3
