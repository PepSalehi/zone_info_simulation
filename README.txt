python run.py -m '1,1.5,2' -f '2500' -p '1.0'

python run.py -m '1,1.5,2' -f '2500' -p '0,0.2,0.4,0.6' 

SEEMS LIKE THE VALUES SHOULD NOT BE A STRING
python run.py -m 1 -f 2500 -d 0
python run.py -m '1' -f '2500' -d 0

for i in 0.0 0.2 0.4 0.6 0.8 1.0; do  python run_parallel.py -k "$i" -r 5 &  done

cat ./Outputs/somefile.csv | csvsort -c prob| csvlook | head
cat ./Outputs/somefile.csv | csvsort --reverse -c total_pickup | csvlook | head -n 100
for f in *.csv; do mv "$f" "${f%.csv}_beta_001.csv";done # https://unix.stackexchange.com/questions/370313/add-text-to-end-of-the-filename-but-before-extension
for i in 0.0 0.2 0.4 0.6 0.8 1.0 ; do for b in 0.1 1 ; do  python run_parallel_for_avg_fare.py -k "$i" -r 15 -bb "$b" & done ; done

for i in 0.0 0.2 0.4 0.6 0.8 1.0 ; do for f in 1500 2000 2500 ; do  python run_parallel_for_avg_fare.py -k "$i" -r 10 -f "$f" & done ; done

for pros in 0 500 1000 1500 2000 ; do for naive in 2000 1500 1000 500; do for do_opt in 'yes' 'no'; do python run_multiple_days.py -PRO "$pros" -NAIVE "$naive" -BH "$do_opt" & done;done;done

for pros in 0 500 1000 ; do for do_opt in 'yes' 'no'; do python run_multiple_days.py -PRO "$pros"  -BH "$do_opt" & done;done

for pros in 0 500 1000 ; do for do_opt in 'yes' 'no'; do python run_multiple_days.py -PRO "$pros"  -BH "$do_opt" & done;done
for pros in 0 500 1000 ; do for s in 'yes'; do python run_multiple_days.py -PRO "$pros"  -SURGE "$s" & done;done

for pros in 0  ; do for do_opt in 'yes' ; do python run_multiple_days.py -PRO "$pros"  -BH "$do_opt" & done;done

for pros in 0 1000 2000 3000 4000 5000; do python run_multiple_days.py -PRO "$pros"  & done
for pros in 0 ; do for LOWER_BOUND_SI in 0.1 0.3 0.5 ; do python run_multiple_days.py -PRO "$pros" -lb "$LOWER_BOUND_SI" & done;done

for pros in 0 1000 2000 3000 4000 5000; do python run_multiple_days.py -PRO "$pros" -info "area_wide" & done

for pros in 1000 ; do for theta in 0.5 1 1.5 ; do python run_multiple_days.py -PRO "$pros" -BH "False" --THETA_prof "$theta"  & done;done

pip install snakeviz
python -m cProfile -s tottime -o myscript.cprof run.py -f 10 -k 0.2
python -m cProfile -s tottime -o myscript4.cprof run_parallel_for_avg_fare.py -f 1000 -k 0.2 -r 1

snakeviz myscript.cprof 
line_profiler
kernprof -l -v run.py -f 10 -k 0.2

<Project root>→/home/peymano/zone_info_simulation; <Project root>→/home/peymano/zone_info_simulation/home/peymano/zone_info_simulation