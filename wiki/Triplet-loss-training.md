Trang nay mo ta cach huan luyen mo hinh Inception Resnet V1 bang cach su dung triplet loss. Tuy nhien, can luu y rang, viec dao tao voi triplet loss se phuc tap hon so voi viec dao tao mo hinh voi softmax. Nhung khi tap huan luyen cua ban co khoang hon 100.000 class thi so luong lop cuoi cung va ban than ham softmax se tro len qua lon, va khi do viec su dung triplet loss se dat hieu qua cao hon. Can luu y trong huong dan nay khong co nghia la cong thuc cuoi cung trong viec dao tao mot mo hinh su dung triplet loss ma day duoc coi la mot cong viec dang duoc tien trien.

De dao tao mot mo hinh voi hieu suat tot hon, vui long tham khao [Classifier training of Inception-ResNet-v1](Classifier-training-of-inception-resnet-v1.md).

## 1. Install Tensorflow
Phien ban hien tai ma chung toi dang su dung cho viec trien khai FaceNet nay do la phien ban Tensorflow v1. No co the duoc cai dat bang cach su dung [pip](https://www.tensorflow.org/get_started/os_setup#pip_installation) hoac tu nguon [sources](https://www.tensorflow.org/get_started/os_setup#installing_from_sources).<br>
Do viec dao tao mang than kinh sau can yeu cau muc do phuc tap tinh toan rat cao, cho nen chung toi se su dung GPU duoc ho tro CUDA. Trang cai dat Tensorflow cung co mot mo ta chi tiet ve cach cai dat tensorflow su dung gpu voi ho tro cua CUDA.

## 2. Clone the FaceNet [repo](https://github.com/davidsandberg/facenet.git)
Dieu nay se duoc thuc hien boi cau lenh: <br>
`git clone https://github.com/davidsandberg/facenet.git`

## 3. Set the python paths
Dat bien moi truong `PYTHONPATH` tro den thu muc `src` cua repo duoc clone nay. Dieu nay duoc thuc hien bang cau lenh: <br>
`export PYTHONPATH=[...]/facenet/src`<br>
trong do `[...]` nen duoc thay the bang duong dan den thu muc noi ban dat repo facnet cua ban.

## 4. Chuan bi cac tap du lieu
### Cau truc cua tap du lieu
Gia dinh rang tap du lieu huan luyen duoc sap xep nhu ben duoi, tuc la trong moi thu muc la danh sach cac hinh anh thuoc mot class nao do:

    Aaron_Eckhart
        Aaron_Eckhart_0001.jpg

    Aaron_Guiel
        Aaron_Guiel_0001.jpg

    Aaron_Patterson
        Aaron_Patterson_0001.jpg

    Aaron_Peirsol
        Aaron_Peirsol_0001.jpg
        Aaron_Peirsol_0002.jpg
        Aaron_Peirsol_0003.jpg
        Aaron_Peirsol_0004.jpg
        ...

### Face alignment
De can chinh khuon mat chung toi su dung mo hinh [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) da duoc chung minh la mang lai hieu suat rat tot cho viec can chinh du lieu trong bo train va bo test. Tren github co mot so ma nguon cung cap viec trien khai MTCNN tren cac nen tang Matlab va Caffe. Ngoai ra ban co the tim thay tap lenh matlab de can chinh bo du lieu bang cach su dung trien khai nay: [here](https://github.com/davidsandberg/facenet/blob/master/tmp/align_dataset.m).

De don gian hoa cho viec su dung trong du an nay, mot trien khai MTCNN tren python/tensorflow duoc cung cap o day: [provided](https://github.com/davidsandberg/facenet/tree/master/src/align). Viec trien khai nay khong co bat ky su phu thuoc nao khac ngoai tensorflow va thoi gian chay tren lfw tuong tu nhu viec trien khai cong viec nay tren matlab.

`python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/  ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44`

Hinh anh dau ra duoc thu nho lai voi kich thuoc la 182 * 182. Dau vao cua mo hinh Inception Resnet v1 la 160 * 160 cung cap mot so margin de co the su dung random crop.

Doi voi cac thu nghiem da duoc thuc hien tren mo hinh Inception Resnet v1, mot margin bo sung duoc them vao la 32 pixel da duoc cong vao. Ly do cho viec bo sung nay la viec mo rong cac khung gioi han bbx se cung cap cho CNN mot so thong tin bo sung ngu canh. Tuy nhien viec cai dat cac tham so nay van chua duoc nghien cuu va rat co the la cac gia tri margin khac co the dan den hieu suat tot hon.

De tang toc qua trinh can chinh, lenh tren co the thuc hien dua vao viec thuc hien song song nhieu tien trinh. Duoi day, cung mot lenh co the chay duoc song song tren 4 process/ De gioi han muc do su dung bo nho cua moi session Tensorflow, tham so `gpu_memory_fraction` se duoc dat thanh 0.25, dieu do co nghia la moi session co the duoc phep su dung toi da 25% bo nho GPU. Co gang lam giam so luong cac tien trinh song song va tang ty le bo nho GPU duoc su dung cho moi session neu dieu nay khien bo nho GPU cua ban bi het:
`for N in {1..4}; do python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/  ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.25 & done`

## 4. Bat dau dao tao
Viec dao tao duoc thuc hien bang cach chay `train_tripletloss.py`. <br>
`python src/train_tripletloss.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate 0.01 --weight_decay 1e-4 --max_nrof_epochs 500`

Khi viec dao tao bat dau duoc thuc hien thi cac thu muc con cho phien dao tao se duoc dat ten theo dinh dang:  `yyyymmdd-hhmm` se duoc tao ra trong thu muc `log_base_dir` va `models_base_dir`. Tham so `data_dir` duoc su dung de chi den vi tri cua tap du lieu huan luyen. Can luu y rang, su ket hop cua mot so bo du lieu co the duoc su dung bang cach tach cac duong dan bang dau `..` Cuoi cung, mo ta mang suy luan duoc dua ra boi tham so `model_def`. Trong vi du tren, `models.inception_resnet_v1` chi den module `inception_resnet_v1`  trong package `models`. Module nay xac dinh function `inference(images, ...)`, trong do `images` la mot placeholder cho hinh anh dau vao (co kich thuoc <?,160,160,3>) va tra ve mot tham chieu den bien `embeddings`.

Neu tham so  `lfw_dir` duoc thiet lap de tro den mot thu muc co so cua bo du lieu LFW, mo hinh se duoc danh gia tren LFW cu sau 1000 batches. De biet thong tin chi tiet cho viec danh gia mot model tren tap du lieu LFW, vui long tham khao: [Validate-on-LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW). Neu khong co tham so nao lien quan den viec danh gia tren bo du lieu LFW trong qua trinh dao tao, thi ban co the de trong tham so `lfw_dir`. Tuy nhien xin luu y rang, tap du lieu LFW o day phai duoc can chinh giong nhu tren tap du lieu huan luyen.

## 5. Running TensorBoard
Trong qua trinh viec dao tao FaceNet dang duoc dien ra, se that thu vi de theo doi qua trinh hoc tap nay dang duoc dien ra nhu the nao. Dieu nay co the duoc thuc hien dua vao [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). De khoi dong TensorBoard, ban co the chay lenh sau:  <br>`tensorboard --logdir=~/logs/facenet --port 6006`<br> va sau do tro den dia chi: <br>http://localhost:6006/ tren trinh duyet yeu thich cua minh.
