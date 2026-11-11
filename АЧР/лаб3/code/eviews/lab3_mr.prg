' ЛР-3. Частина 3: кореляційна матриця та рівняння множинної регресії для rts1
' Аргумент: %0 - папка з даними (без пробілів, із "\" в кінці)
' Приклад: run lab3_mr d:\ats_lab_03_new\data\

wfcreate(wf=lab3_mr) u 178
read(t=txt) {%0}rts1.txt rts1
for %s rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl
  read(t=txt) {%0}{%s}.txt {%s}
next

' ---------- Кореляційна матриця індексів ----------
group g_all rts1 rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl
freeze(tab_cor) g_all.cor
show tab_cor

' ---------- Регресори з |r(rts1, x)| >= 0.5 ----------
%regs = ""
for %s rtscr rtseu rtsfn rtsin rtsmm rtsog rtstl
  if @abs(@cor(rts1, {%s})) >= 0.5 then
    %regs = %regs + " " + %s
  endif
next

' ---------- Рівняння множинної регресії та послідовне вилучення незначущих регресорів ----------
equation eq_mr1.ls rts1 c {%regs}
equation eq_mr2.ls rts1 c rtscr rtsfn rtsin rtsmm rtsog rtstl
equation eq_mr3.ls rts1 c rtsfn rtsin rtsmm rtsog rtstl
show eq_mr1
show eq_mr3
