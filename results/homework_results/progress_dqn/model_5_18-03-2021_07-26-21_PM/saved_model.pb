»µ
æ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8ś§

dqn_17/dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namedqn_17/dense_68/kernel

*dqn_17/dense_68/kernel/Read/ReadVariableOpReadVariableOpdqn_17/dense_68/kernel*
_output_shapes

:*
dtype0

dqn_17/dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedqn_17/dense_68/bias
y
(dqn_17/dense_68/bias/Read/ReadVariableOpReadVariableOpdqn_17/dense_68/bias*
_output_shapes
:*
dtype0

dqn_17/dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_namedqn_17/dense_69/kernel

*dqn_17/dense_69/kernel/Read/ReadVariableOpReadVariableOpdqn_17/dense_69/kernel*
_output_shapes

: *
dtype0

dqn_17/dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namedqn_17/dense_69/bias
y
(dqn_17/dense_69/bias/Read/ReadVariableOpReadVariableOpdqn_17/dense_69/bias*
_output_shapes
: *
dtype0

dqn_17/dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_namedqn_17/dense_70/kernel

*dqn_17/dense_70/kernel/Read/ReadVariableOpReadVariableOpdqn_17/dense_70/kernel*
_output_shapes

:  *
dtype0

dqn_17/dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namedqn_17/dense_70/bias
y
(dqn_17/dense_70/bias/Read/ReadVariableOpReadVariableOpdqn_17/dense_70/bias*
_output_shapes
: *
dtype0

dqn_17/dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_namedqn_17/dense_71/kernel

*dqn_17/dense_71/kernel/Read/ReadVariableOpReadVariableOpdqn_17/dense_71/kernel*
_output_shapes

: *
dtype0

dqn_17/dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namedqn_17/dense_71/bias
y
(dqn_17/dense_71/bias/Read/ReadVariableOpReadVariableOpdqn_17/dense_71/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ė
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueüBł Bņ

d1
d2
d3
dout
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
x


activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
x

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
x

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
8
0
1
2
3
4
5
6
 7
8
0
1
2
3
4
5
6
 7
 
­
%non_trainable_variables

&layers
	variables
trainable_variables
regularization_losses
'layer_metrics
(metrics
)layer_regularization_losses
 
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
PN
VARIABLE_VALUEdqn_17/dense_68/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdqn_17/dense_68/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
.non_trainable_variables

/layers
	variables
trainable_variables
regularization_losses
0layer_metrics
1metrics
2layer_regularization_losses
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
PN
VARIABLE_VALUEdqn_17/dense_69/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdqn_17/dense_69/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
7non_trainable_variables

8layers
	variables
trainable_variables
regularization_losses
9layer_metrics
:metrics
;layer_regularization_losses
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
PN
VARIABLE_VALUEdqn_17/dense_70/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdqn_17/dense_70/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
@non_trainable_variables

Alayers
	variables
trainable_variables
regularization_losses
Blayer_metrics
Cmetrics
Dlayer_regularization_losses
RP
VARIABLE_VALUEdqn_17/dense_71/kernel&dout/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_17/dense_71/bias$dout/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
­
Enon_trainable_variables

Flayers
!	variables
"trainable_variables
#regularization_losses
Glayer_metrics
Hmetrics
Ilayer_regularization_losses
 

0
1
2
3
 
 
 
 
 
 
­
Jnon_trainable_variables

Klayers
*	variables
+trainable_variables
,regularization_losses
Llayer_metrics
Mmetrics
Nlayer_regularization_losses
 


0
 
 
 
 
 
 
­
Onon_trainable_variables

Players
3	variables
4trainable_variables
5regularization_losses
Qlayer_metrics
Rmetrics
Slayer_regularization_losses
 

0
 
 
 
 
 
 
­
Tnon_trainable_variables

Ulayers
<	variables
=trainable_variables
>regularization_losses
Vlayer_metrics
Wmetrics
Xlayer_regularization_losses
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
ś
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn_17/dense_68/kerneldqn_17/dense_68/biasdqn_17/dense_69/kerneldqn_17/dense_69/biasdqn_17/dense_70/kerneldqn_17/dense_70/biasdqn_17/dense_71/kerneldqn_17/dense_71/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_267559
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ū
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*dqn_17/dense_68/kernel/Read/ReadVariableOp(dqn_17/dense_68/bias/Read/ReadVariableOp*dqn_17/dense_69/kernel/Read/ReadVariableOp(dqn_17/dense_69/bias/Read/ReadVariableOp*dqn_17/dense_70/kernel/Read/ReadVariableOp(dqn_17/dense_70/bias/Read/ReadVariableOp*dqn_17/dense_71/kernel/Read/ReadVariableOp(dqn_17/dense_71/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_267685
Ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_17/dense_68/kerneldqn_17/dense_68/biasdqn_17/dense_69/kerneldqn_17/dense_69/biasdqn_17/dense_70/kerneldqn_17/dense_70/biasdqn_17/dense_71/kerneldqn_17/dense_71/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_267719¾ī
ß

__inference__traced_save_267685
file_prefix5
1savev2_dqn_17_dense_68_kernel_read_readvariableop3
/savev2_dqn_17_dense_68_bias_read_readvariableop5
1savev2_dqn_17_dense_69_kernel_read_readvariableop3
/savev2_dqn_17_dense_69_bias_read_readvariableop5
1savev2_dqn_17_dense_70_kernel_read_readvariableop3
/savev2_dqn_17_dense_70_bias_read_readvariableop5
1savev2_dqn_17_dense_71_kernel_read_readvariableop3
/savev2_dqn_17_dense_71_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3e61d8bacc6942a08135710ba0c5fdce/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĶ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ß
valueÕBŅ	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB&dout/kernel/.ATTRIBUTES/VARIABLE_VALUEB$dout/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesŅ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_dqn_17_dense_68_kernel_read_readvariableop/savev2_dqn_17_dense_68_bias_read_readvariableop1savev2_dqn_17_dense_69_kernel_read_readvariableop/savev2_dqn_17_dense_69_bias_read_readvariableop1savev2_dqn_17_dense_70_kernel_read_readvariableop/savev2_dqn_17_dense_70_bias_read_readvariableop1savev2_dqn_17_dense_71_kernel_read_readvariableop/savev2_dqn_17_dense_71_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: ::: : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: 
ū
¬
D__inference_dense_70_layer_call_and_return_conditional_losses_267610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAdd
leaky_re_lu_53/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_53/LeakyReluz
IdentityIdentity&leaky_re_lu_53/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ķ
¬
D__inference_dense_71_layer_call_and_return_conditional_losses_267497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
É%
Ü
"__inference__traced_restore_267719
file_prefix+
'assignvariableop_dqn_17_dense_68_kernel+
'assignvariableop_1_dqn_17_dense_68_bias-
)assignvariableop_2_dqn_17_dense_69_kernel+
'assignvariableop_3_dqn_17_dense_69_bias-
)assignvariableop_4_dqn_17_dense_70_kernel+
'assignvariableop_5_dqn_17_dense_70_bias-
)assignvariableop_6_dqn_17_dense_71_kernel+
'assignvariableop_7_dqn_17_dense_71_bias

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7Ó
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ß
valueÕBŅ	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB&dout/kernel/.ATTRIBUTES/VARIABLE_VALUEB$dout/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesŲ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¦
AssignVariableOpAssignVariableOp'assignvariableop_dqn_17_dense_68_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¬
AssignVariableOp_1AssignVariableOp'assignvariableop_1_dqn_17_dense_68_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp)assignvariableop_2_dqn_17_dense_69_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¬
AssignVariableOp_3AssignVariableOp'assignvariableop_3_dqn_17_dense_69_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp)assignvariableop_4_dqn_17_dense_70_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¬
AssignVariableOp_5AssignVariableOp'assignvariableop_5_dqn_17_dense_70_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp)assignvariableop_6_dqn_17_dense_71_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¬
AssignVariableOp_7AssignVariableOp'assignvariableop_7_dqn_17_dense_71_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ū
¬
D__inference_dense_69_layer_call_and_return_conditional_losses_267444

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAdd
leaky_re_lu_52/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_52/LeakyReluz
IdentityIdentity&leaky_re_lu_52/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ū
¬
D__inference_dense_70_layer_call_and_return_conditional_losses_267471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAdd
leaky_re_lu_53/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_53/LeakyReluz
IdentityIdentity&leaky_re_lu_53/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
ū
¬
D__inference_dense_68_layer_call_and_return_conditional_losses_267570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
leaky_re_lu_51/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_51/LeakyReluz
IdentityIdentity&leaky_re_lu_51/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼

B__inference_dqn_17_layer_call_and_return_conditional_losses_267514
input_1
dense_68_267428
dense_68_267430
dense_69_267455
dense_69_267457
dense_70_267482
dense_70_267484
dense_71_267508
dense_71_267510
identity¢ dense_68/StatefulPartitionedCall¢ dense_69/StatefulPartitionedCall¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall
 dense_68/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_68_267428dense_68_267430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_2674172"
 dense_68/StatefulPartitionedCall·
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_267455dense_69_267457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_2674442"
 dense_69/StatefulPartitionedCall·
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_267482dense_70_267484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_2674712"
 dense_70/StatefulPartitionedCall·
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_267508dense_71_267510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_2674972"
 dense_71/StatefulPartitionedCall
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ü
~
)__inference_dense_71_layer_call_fn_267638

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_2674972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
ū
¬
D__inference_dense_69_layer_call_and_return_conditional_losses_267590

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
BiasAdd
leaky_re_lu_52/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_52/LeakyReluz
IdentityIdentity&leaky_re_lu_52/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü
~
)__inference_dense_69_layer_call_fn_267599

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_2674442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ū
¬
D__inference_dense_68_layer_call_and_return_conditional_losses_267417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdd
leaky_re_lu_51/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_51/LeakyReluz
IdentityIdentity&leaky_re_lu_51/LeakyRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ü
~
)__inference_dense_68_layer_call_fn_267579

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_2674172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

×
'__inference_dqn_17_layer_call_fn_267536
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĮ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dqn_17_layer_call_and_return_conditional_losses_2675142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
ų
Ō
$__inference_signature_wrapper_267559
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_2674022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ķ
¬
D__inference_dense_71_layer_call_and_return_conditional_losses_267629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ü
~
)__inference_dense_70_layer_call_fn_267619

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_2674712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
¤#
å
!__inference__wrapped_model_267402
input_12
.dqn_17_dense_68_matmul_readvariableop_resource3
/dqn_17_dense_68_biasadd_readvariableop_resource2
.dqn_17_dense_69_matmul_readvariableop_resource3
/dqn_17_dense_69_biasadd_readvariableop_resource2
.dqn_17_dense_70_matmul_readvariableop_resource3
/dqn_17_dense_70_biasadd_readvariableop_resource2
.dqn_17_dense_71_matmul_readvariableop_resource3
/dqn_17_dense_71_biasadd_readvariableop_resource
identity½
%dqn_17/dense_68/MatMul/ReadVariableOpReadVariableOp.dqn_17_dense_68_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%dqn_17/dense_68/MatMul/ReadVariableOp¤
dqn_17/dense_68/MatMulMatMulinput_1-dqn_17/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_17/dense_68/MatMul¼
&dqn_17/dense_68/BiasAdd/ReadVariableOpReadVariableOp/dqn_17_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&dqn_17/dense_68/BiasAdd/ReadVariableOpĮ
dqn_17/dense_68/BiasAddBiasAdd dqn_17/dense_68/MatMul:product:0.dqn_17/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_17/dense_68/BiasAddÅ
(dqn_17/dense_68/leaky_re_lu_51/LeakyRelu	LeakyRelu dqn_17/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2*
(dqn_17/dense_68/leaky_re_lu_51/LeakyRelu½
%dqn_17/dense_69/MatMul/ReadVariableOpReadVariableOp.dqn_17_dense_69_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%dqn_17/dense_69/MatMul/ReadVariableOpÓ
dqn_17/dense_69/MatMulMatMul6dqn_17/dense_68/leaky_re_lu_51/LeakyRelu:activations:0-dqn_17/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_17/dense_69/MatMul¼
&dqn_17/dense_69/BiasAdd/ReadVariableOpReadVariableOp/dqn_17_dense_69_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&dqn_17/dense_69/BiasAdd/ReadVariableOpĮ
dqn_17/dense_69/BiasAddBiasAdd dqn_17/dense_69/MatMul:product:0.dqn_17/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_17/dense_69/BiasAddÅ
(dqn_17/dense_69/leaky_re_lu_52/LeakyRelu	LeakyRelu dqn_17/dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2*
(dqn_17/dense_69/leaky_re_lu_52/LeakyRelu½
%dqn_17/dense_70/MatMul/ReadVariableOpReadVariableOp.dqn_17_dense_70_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02'
%dqn_17/dense_70/MatMul/ReadVariableOpÓ
dqn_17/dense_70/MatMulMatMul6dqn_17/dense_69/leaky_re_lu_52/LeakyRelu:activations:0-dqn_17/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_17/dense_70/MatMul¼
&dqn_17/dense_70/BiasAdd/ReadVariableOpReadVariableOp/dqn_17_dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&dqn_17/dense_70/BiasAdd/ReadVariableOpĮ
dqn_17/dense_70/BiasAddBiasAdd dqn_17/dense_70/MatMul:product:0.dqn_17/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_17/dense_70/BiasAddÅ
(dqn_17/dense_70/leaky_re_lu_53/LeakyRelu	LeakyRelu dqn_17/dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2*
(dqn_17/dense_70/leaky_re_lu_53/LeakyRelu½
%dqn_17/dense_71/MatMul/ReadVariableOpReadVariableOp.dqn_17_dense_71_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%dqn_17/dense_71/MatMul/ReadVariableOpÓ
dqn_17/dense_71/MatMulMatMul6dqn_17/dense_70/leaky_re_lu_53/LeakyRelu:activations:0-dqn_17/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_17/dense_71/MatMul¼
&dqn_17/dense_71/BiasAdd/ReadVariableOpReadVariableOp/dqn_17_dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&dqn_17/dense_71/BiasAdd/ReadVariableOpĮ
dqn_17/dense_71/BiasAddBiasAdd dqn_17/dense_71/MatMul:product:0.dqn_17/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_17/dense_71/BiasAddt
IdentityIdentity dqn_17/dense_71/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::::P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0’’’’’’’’’<
q_values0
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:
Ģ
d1
d2
d3
dout
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
[__call__"ī
_tf_keras_modelŌ{"class_name": "DQN", "name": "dqn_17", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DQN"}}
ķ


activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"ø
_tf_keras_layer{"class_name": "Dense", "name": "dense_68", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_68", "trainable": true, "dtype": "float64", "units": 16, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_51", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
ļ

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*^&call_and_return_all_conditional_losses
___call__"ŗ
_tf_keras_layer {"class_name": "Dense", "name": "dense_69", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_69", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_52", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
ļ

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"ŗ
_tf_keras_layer {"class_name": "Dense", "name": "dense_70", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_70", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_53", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
š

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"Ė
_tf_keras_layer±{"class_name": "Dense", "name": "dense_71", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_71", "trainable": true, "dtype": "float64", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
%non_trainable_variables

&layers
	variables
trainable_variables
regularization_losses
'layer_metrics
(metrics
)layer_regularization_losses
[__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
dserving_default"
signature_map
Š
*	variables
+trainable_variables
,regularization_losses
-	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Į
_tf_keras_layer§{"class_name": "LeakyReLU", "name": "leaky_re_lu_51", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_51", "trainable": true, "dtype": "float64", "alpha": 0.3}}
(:&2dqn_17/dense_68/kernel
": 2dqn_17/dense_68/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.non_trainable_variables

/layers
	variables
trainable_variables
regularization_losses
0layer_metrics
1metrics
2layer_regularization_losses
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Š
3	variables
4trainable_variables
5regularization_losses
6	keras_api
*g&call_and_return_all_conditional_losses
h__call__"Į
_tf_keras_layer§{"class_name": "LeakyReLU", "name": "leaky_re_lu_52", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_52", "trainable": true, "dtype": "float64", "alpha": 0.3}}
(:& 2dqn_17/dense_69/kernel
":  2dqn_17/dense_69/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
	variables
trainable_variables
regularization_losses
9layer_metrics
:metrics
;layer_regularization_losses
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Š
<	variables
=trainable_variables
>regularization_losses
?	keras_api
*i&call_and_return_all_conditional_losses
j__call__"Į
_tf_keras_layer§{"class_name": "LeakyReLU", "name": "leaky_re_lu_53", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_53", "trainable": true, "dtype": "float64", "alpha": 0.3}}
(:&  2dqn_17/dense_70/kernel
":  2dqn_17/dense_70/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
@non_trainable_variables

Alayers
	variables
trainable_variables
regularization_losses
Blayer_metrics
Cmetrics
Dlayer_regularization_losses
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
(:& 2dqn_17/dense_71/kernel
": 2dqn_17/dense_71/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
!	variables
"trainable_variables
#regularization_losses
Glayer_metrics
Hmetrics
Ilayer_regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
*	variables
+trainable_variables
,regularization_losses
Llayer_metrics
Mmetrics
Nlayer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
3	variables
4trainable_variables
5regularization_losses
Qlayer_metrics
Rmetrics
Slayer_regularization_losses
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
<	variables
=trainable_variables
>regularization_losses
Vlayer_metrics
Wmetrics
Xlayer_regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ß2Ü
!__inference__wrapped_model_267402¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
2
B__inference_dqn_17_layer_call_and_return_conditional_losses_267514Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ō2ń
'__inference_dqn_17_layer_call_fn_267536Å
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *&¢#
!
input_1’’’’’’’’’
ī2ė
D__inference_dense_68_layer_call_and_return_conditional_losses_267570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_68_layer_call_fn_267579¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_69_layer_call_and_return_conditional_losses_267590¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_69_layer_call_fn_267599¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_70_layer_call_and_return_conditional_losses_267610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_70_layer_call_fn_267619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_71_layer_call_and_return_conditional_losses_267629¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_71_layer_call_fn_267638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
3B1
$__inference_signature_wrapper_267559input_1
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
!__inference__wrapped_model_267402q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’¤
D__inference_dense_68_layer_call_and_return_conditional_losses_267570\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 |
)__inference_dense_68_layer_call_fn_267579O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¤
D__inference_dense_69_layer_call_and_return_conditional_losses_267590\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’ 
 |
)__inference_dense_69_layer_call_fn_267599O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ ¤
D__inference_dense_70_layer_call_and_return_conditional_losses_267610\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’ 
 |
)__inference_dense_70_layer_call_fn_267619O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’ ¤
D__inference_dense_71_layer_call_and_return_conditional_losses_267629\ /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 |
)__inference_dense_71_layer_call_fn_267638O /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’Ć
B__inference_dqn_17_layer_call_and_return_conditional_losses_267514} 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "?¢<
5Ŗ2
0
q_values$!

0/q_values’’’’’’’’’
 
'__inference_dqn_17_layer_call_fn_267536q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’¤
$__inference_signature_wrapper_267559| ;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"3Ŗ0
.
q_values"
q_values’’’’’’’’’