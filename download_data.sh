
# download train and test ds
wget https://www.dropbox.com/scl/fi/rdgr7b3z795qgebfso158/train-100.zip?rlkey=v8hq67y6ln1s4oaq714dfwz5g&dl=0
wget https://www.dropbox.com/scl/fi/ipxruod2dpv8tel0bz8c1/test.zip?rlkey=172pmq41qyhzpmvav2gy89k27&dl=0


unzip train-100.zip?rlkey=v8hq67y6ln1s4oaq714dfwz5g
unzip test.zip?rlkey=172pmq41qyhzpmvav2gy89k27

mkdir libri100min

cp -r test libri100min
rm -rf test

cp -r train-100 libri100min
rm -rf  train-100

mv libri100min/train-100  libri100min/train

mv  libri100min/train/mix_clean libri100min/train/mix
mv  libri100min/train/s1 libri100min/train/target
mv  libri100min/test/mix_clean libri100min/test/mix
mv  libri100min/test/s1 libri100min/test/target

# download overfit ds

wget https://www.dropbox.com/scl/fi/a2juz4usbliwufnimrz9k/overfit_ds.zip?rlkey=3atk8ntb1h2bqhdsuwvg2le7l&dl=0
unzip overfit_ds.zip?rlkey=3atk8ntb1h2bqhdsuwvg2le7l