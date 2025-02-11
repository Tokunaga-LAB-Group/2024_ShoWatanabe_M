function exec()
{
	echo ======== No.$3 $1 $2 =======
	# make annotation_multi_$6 LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
    # make cospa LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
	# make c_test LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
}

function exec-pointnet()
{
	echo ======== No.$3 $1 $2 =======
	# make annotation_multi LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3
    make pointnet LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100
	make p_test LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100
}
for cate in Airplane
# for cate in Car Chair Earphone Knife Lamp Motorbike Skateboard Table
do
	# for num in 0 1 2 3 4 5 6 7 8 9
	for num in 0 1 2 3 4
	do
		for ap in 1
		# for ap in 0.01 0.1 0.3 0.5 0.7 0.9 1
		# for ap in 1
		do
			for lr in 1e-4
			do
				for type in const
				do
					if [ $ap -eq 1 ]; then
						exec-pointnet $lr $ap $num 2 $cate $type
					else
						exec $lr $ap $num 2 $cate $type
					fi
				done
			done
		done
	done
done
