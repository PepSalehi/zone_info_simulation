python run.py -m '1,1.5,2' -f '2500' -p '1.0'

python run.py -m '1,1.5,2' -f '2500' -p '0,0.2,0.4,0.6' 

SEEMS LIKE THE VALUES SHOULD NOT BE A STRING
python run.py -m 1 -f 2500 -d 0
python run.py -m '1' -f '2500' -d 0

for i in 0.0 0.2 0.4 0.6 0.8 1.0; do  python run_parallel.py -k "$i" -r 5 &  done

cat ./Outputs/somefile.csv | csvsort -c prob| csvlook | head
cat ./Outputs/somefile.csv | csvsort --reverse -c total_pickup | csvlook | head -n 100
for f in *.csv; do mv "$f" "${f%.csv}_beta_001.csv";done # https://unix.stackexchange.com/questions/370313/add-text-to-end-of-the-filename-but-before-extension
