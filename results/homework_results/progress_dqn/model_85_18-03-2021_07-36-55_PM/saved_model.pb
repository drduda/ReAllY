ģ¹
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
 "serve*2.3.02unknown8Ż«

dqn_213/dense_852/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namedqn_213/dense_852/kernel

,dqn_213/dense_852/kernel/Read/ReadVariableOpReadVariableOpdqn_213/dense_852/kernel*
_output_shapes

:*
dtype0

dqn_213/dense_852/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_213/dense_852/bias
}
*dqn_213/dense_852/bias/Read/ReadVariableOpReadVariableOpdqn_213/dense_852/bias*
_output_shapes
:*
dtype0

dqn_213/dense_853/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_213/dense_853/kernel

,dqn_213/dense_853/kernel/Read/ReadVariableOpReadVariableOpdqn_213/dense_853/kernel*
_output_shapes

: *
dtype0

dqn_213/dense_853/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_213/dense_853/bias
}
*dqn_213/dense_853/bias/Read/ReadVariableOpReadVariableOpdqn_213/dense_853/bias*
_output_shapes
: *
dtype0

dqn_213/dense_854/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_namedqn_213/dense_854/kernel

,dqn_213/dense_854/kernel/Read/ReadVariableOpReadVariableOpdqn_213/dense_854/kernel*
_output_shapes

:  *
dtype0

dqn_213/dense_854/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_213/dense_854/bias
}
*dqn_213/dense_854/bias/Read/ReadVariableOpReadVariableOpdqn_213/dense_854/bias*
_output_shapes
: *
dtype0

dqn_213/dense_855/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_213/dense_855/kernel

,dqn_213/dense_855/kernel/Read/ReadVariableOpReadVariableOpdqn_213/dense_855/kernel*
_output_shapes

: *
dtype0

dqn_213/dense_855/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_213/dense_855/bias
}
*dqn_213/dense_855/bias/Read/ReadVariableOpReadVariableOpdqn_213/dense_855/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ū
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
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
RP
VARIABLE_VALUEdqn_213/dense_852/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_213/dense_852/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
RP
VARIABLE_VALUEdqn_213/dense_853/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_213/dense_853/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
RP
VARIABLE_VALUEdqn_213/dense_854/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_213/dense_854/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
TR
VARIABLE_VALUEdqn_213/dense_855/kernel&dout/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdqn_213/dense_855/bias$dout/bias/.ATTRIBUTES/VARIABLE_VALUE
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

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn_213/dense_852/kerneldqn_213/dense_852/biasdqn_213/dense_853/kerneldqn_213/dense_853/biasdqn_213/dense_854/kerneldqn_213/dense_854/biasdqn_213/dense_855/kerneldqn_213/dense_855/bias*
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4362627
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,dqn_213/dense_852/kernel/Read/ReadVariableOp*dqn_213/dense_852/bias/Read/ReadVariableOp,dqn_213/dense_853/kernel/Read/ReadVariableOp*dqn_213/dense_853/bias/Read/ReadVariableOp,dqn_213/dense_854/kernel/Read/ReadVariableOp*dqn_213/dense_854/bias/Read/ReadVariableOp,dqn_213/dense_855/kernel/Read/ReadVariableOp*dqn_213/dense_855/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_4362753
ē
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_213/dense_852/kerneldqn_213/dense_852/biasdqn_213/dense_853/kerneldqn_213/dense_853/biasdqn_213/dense_854/kerneldqn_213/dense_854/biasdqn_213/dense_855/kerneldqn_213/dense_855/bias*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_4362787ń
 
Ł
)__inference_dqn_213_layer_call_fn_4362604
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĆ
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
GPU 2J 8 *M
fHRF
D__inference_dqn_213_layer_call_and_return_conditional_losses_43625822
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
	
®
F__inference_dense_854_layer_call_and_return_conditional_losses_4362539

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
BiasAdd
leaky_re_lu_641/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_641/LeakyRelu{
IdentityIdentity'leaky_re_lu_641/LeakyRelu:activations:0*
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
Ļ
®
F__inference_dense_855_layer_call_and_return_conditional_losses_4362697

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
	
®
F__inference_dense_853_layer_call_and_return_conditional_losses_4362658

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
BiasAdd
leaky_re_lu_640/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_640/LeakyRelu{
IdentityIdentity'leaky_re_lu_640/LeakyRelu:activations:0*
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
	
®
F__inference_dense_853_layer_call_and_return_conditional_losses_4362512

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
BiasAdd
leaky_re_lu_640/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_640/LeakyRelu{
IdentityIdentity'leaky_re_lu_640/LeakyRelu:activations:0*
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
	
®
F__inference_dense_852_layer_call_and_return_conditional_losses_4362638

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
BiasAdd
leaky_re_lu_639/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_639/LeakyRelu{
IdentityIdentity'leaky_re_lu_639/LeakyRelu:activations:0*
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
ś
Õ
%__inference_signature_wrapper_4362627
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall”
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
GPU 2J 8 *+
f&R$
"__inference__wrapped_model_43624702
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

¬
D__inference_dqn_213_layer_call_and_return_conditional_losses_4362582
input_1
dense_852_4362496
dense_852_4362498
dense_853_4362523
dense_853_4362525
dense_854_4362550
dense_854_4362552
dense_855_4362576
dense_855_4362578
identity¢!dense_852/StatefulPartitionedCall¢!dense_853/StatefulPartitionedCall¢!dense_854/StatefulPartitionedCall¢!dense_855/StatefulPartitionedCall
!dense_852/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_852_4362496dense_852_4362498*
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
GPU 2J 8 *O
fJRH
F__inference_dense_852_layer_call_and_return_conditional_losses_43624852#
!dense_852/StatefulPartitionedCallĄ
!dense_853/StatefulPartitionedCallStatefulPartitionedCall*dense_852/StatefulPartitionedCall:output:0dense_853_4362523dense_853_4362525*
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
GPU 2J 8 *O
fJRH
F__inference_dense_853_layer_call_and_return_conditional_losses_43625122#
!dense_853/StatefulPartitionedCallĄ
!dense_854/StatefulPartitionedCallStatefulPartitionedCall*dense_853/StatefulPartitionedCall:output:0dense_854_4362550dense_854_4362552*
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
GPU 2J 8 *O
fJRH
F__inference_dense_854_layer_call_and_return_conditional_losses_43625392#
!dense_854/StatefulPartitionedCallĄ
!dense_855/StatefulPartitionedCallStatefulPartitionedCall*dense_854/StatefulPartitionedCall:output:0dense_855_4362576dense_855_4362578*
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
GPU 2J 8 *O
fJRH
F__inference_dense_855_layer_call_and_return_conditional_losses_43625652#
!dense_855/StatefulPartitionedCall
IdentityIdentity*dense_855/StatefulPartitionedCall:output:0"^dense_852/StatefulPartitionedCall"^dense_853/StatefulPartitionedCall"^dense_854/StatefulPartitionedCall"^dense_855/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2F
!dense_852/StatefulPartitionedCall!dense_852/StatefulPartitionedCall2F
!dense_853/StatefulPartitionedCall!dense_853/StatefulPartitionedCall2F
!dense_854/StatefulPartitionedCall!dense_854/StatefulPartitionedCall2F
!dense_855/StatefulPartitionedCall!dense_855/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ļ
®
F__inference_dense_855_layer_call_and_return_conditional_losses_4362565

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
į

+__inference_dense_854_layer_call_fn_4362687

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_854_layer_call_and_return_conditional_losses_43625392
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
į

+__inference_dense_855_layer_call_fn_4362706

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_855_layer_call_and_return_conditional_losses_43625652
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
	
®
F__inference_dense_852_layer_call_and_return_conditional_losses_4362485

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
BiasAdd
leaky_re_lu_639/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_639/LeakyRelu{
IdentityIdentity'leaky_re_lu_639/LeakyRelu:activations:0*
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

­
 __inference__traced_save_4362753
file_prefix7
3savev2_dqn_213_dense_852_kernel_read_readvariableop5
1savev2_dqn_213_dense_852_bias_read_readvariableop7
3savev2_dqn_213_dense_853_kernel_read_readvariableop5
1savev2_dqn_213_dense_853_bias_read_readvariableop7
3savev2_dqn_213_dense_854_kernel_read_readvariableop5
1savev2_dqn_213_dense_854_bias_read_readvariableop7
3savev2_dqn_213_dense_855_kernel_read_readvariableop5
1savev2_dqn_213_dense_855_bias_read_readvariableop
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
value3B1 B+_temp_c382701a4ac646338b6be5fdc2eb2916/part2	
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
SaveV2/shape_and_slicesā
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_dqn_213_dense_852_kernel_read_readvariableop1savev2_dqn_213_dense_852_bias_read_readvariableop3savev2_dqn_213_dense_853_kernel_read_readvariableop1savev2_dqn_213_dense_853_bias_read_readvariableop3savev2_dqn_213_dense_854_kernel_read_readvariableop1savev2_dqn_213_dense_854_bias_read_readvariableop3savev2_dqn_213_dense_855_kernel_read_readvariableop1savev2_dqn_213_dense_855_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
	
®
F__inference_dense_854_layer_call_and_return_conditional_losses_4362678

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
BiasAdd
leaky_re_lu_641/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_641/LeakyRelu{
IdentityIdentity'leaky_re_lu_641/LeakyRelu:activations:0*
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
ź%
ķ
#__inference__traced_restore_4362787
file_prefix-
)assignvariableop_dqn_213_dense_852_kernel-
)assignvariableop_1_dqn_213_dense_852_bias/
+assignvariableop_2_dqn_213_dense_853_kernel-
)assignvariableop_3_dqn_213_dense_853_bias/
+assignvariableop_4_dqn_213_dense_854_kernel-
)assignvariableop_5_dqn_213_dense_854_bias/
+assignvariableop_6_dqn_213_dense_855_kernel-
)assignvariableop_7_dqn_213_dense_855_bias

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

IdentityØ
AssignVariableOpAssignVariableOp)assignvariableop_dqn_213_dense_852_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_dqn_213_dense_852_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_dqn_213_dense_853_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp)assignvariableop_3_dqn_213_dense_853_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOp+assignvariableop_4_dqn_213_dense_854_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5®
AssignVariableOp_5AssignVariableOp)assignvariableop_5_dqn_213_dense_854_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOp+assignvariableop_6_dqn_213_dense_855_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7®
AssignVariableOp_7AssignVariableOp)assignvariableop_7_dqn_213_dense_855_biasIdentity_7:output:0"/device:CPU:0*
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
į

+__inference_dense_852_layer_call_fn_4362647

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_852_layer_call_and_return_conditional_losses_43624852
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
į

+__inference_dense_853_layer_call_fn_4362667

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
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
GPU 2J 8 *O
fJRH
F__inference_dense_853_layer_call_and_return_conditional_losses_43625122
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
Ą$
ö
"__inference__wrapped_model_4362470
input_14
0dqn_213_dense_852_matmul_readvariableop_resource5
1dqn_213_dense_852_biasadd_readvariableop_resource4
0dqn_213_dense_853_matmul_readvariableop_resource5
1dqn_213_dense_853_biasadd_readvariableop_resource4
0dqn_213_dense_854_matmul_readvariableop_resource5
1dqn_213_dense_854_biasadd_readvariableop_resource4
0dqn_213_dense_855_matmul_readvariableop_resource5
1dqn_213_dense_855_biasadd_readvariableop_resource
identityĆ
'dqn_213/dense_852/MatMul/ReadVariableOpReadVariableOp0dqn_213_dense_852_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'dqn_213/dense_852/MatMul/ReadVariableOpŖ
dqn_213/dense_852/MatMulMatMulinput_1/dqn_213/dense_852/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_213/dense_852/MatMulĀ
(dqn_213/dense_852/BiasAdd/ReadVariableOpReadVariableOp1dqn_213_dense_852_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(dqn_213/dense_852/BiasAdd/ReadVariableOpÉ
dqn_213/dense_852/BiasAddBiasAdd"dqn_213/dense_852/MatMul:product:00dqn_213/dense_852/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_213/dense_852/BiasAddĶ
+dqn_213/dense_852/leaky_re_lu_639/LeakyRelu	LeakyRelu"dqn_213/dense_852/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2-
+dqn_213/dense_852/leaky_re_lu_639/LeakyReluĆ
'dqn_213/dense_853/MatMul/ReadVariableOpReadVariableOp0dqn_213_dense_853_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'dqn_213/dense_853/MatMul/ReadVariableOpÜ
dqn_213/dense_853/MatMulMatMul9dqn_213/dense_852/leaky_re_lu_639/LeakyRelu:activations:0/dqn_213/dense_853/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_213/dense_853/MatMulĀ
(dqn_213/dense_853/BiasAdd/ReadVariableOpReadVariableOp1dqn_213_dense_853_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(dqn_213/dense_853/BiasAdd/ReadVariableOpÉ
dqn_213/dense_853/BiasAddBiasAdd"dqn_213/dense_853/MatMul:product:00dqn_213/dense_853/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_213/dense_853/BiasAddĶ
+dqn_213/dense_853/leaky_re_lu_640/LeakyRelu	LeakyRelu"dqn_213/dense_853/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2-
+dqn_213/dense_853/leaky_re_lu_640/LeakyReluĆ
'dqn_213/dense_854/MatMul/ReadVariableOpReadVariableOp0dqn_213_dense_854_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'dqn_213/dense_854/MatMul/ReadVariableOpÜ
dqn_213/dense_854/MatMulMatMul9dqn_213/dense_853/leaky_re_lu_640/LeakyRelu:activations:0/dqn_213/dense_854/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_213/dense_854/MatMulĀ
(dqn_213/dense_854/BiasAdd/ReadVariableOpReadVariableOp1dqn_213_dense_854_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(dqn_213/dense_854/BiasAdd/ReadVariableOpÉ
dqn_213/dense_854/BiasAddBiasAdd"dqn_213/dense_854/MatMul:product:00dqn_213/dense_854/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_213/dense_854/BiasAddĶ
+dqn_213/dense_854/leaky_re_lu_641/LeakyRelu	LeakyRelu"dqn_213/dense_854/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2-
+dqn_213/dense_854/leaky_re_lu_641/LeakyReluĆ
'dqn_213/dense_855/MatMul/ReadVariableOpReadVariableOp0dqn_213_dense_855_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'dqn_213/dense_855/MatMul/ReadVariableOpÜ
dqn_213/dense_855/MatMulMatMul9dqn_213/dense_854/leaky_re_lu_641/LeakyRelu:activations:0/dqn_213/dense_855/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_213/dense_855/MatMulĀ
(dqn_213/dense_855/BiasAdd/ReadVariableOpReadVariableOp1dqn_213_dense_855_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(dqn_213/dense_855/BiasAdd/ReadVariableOpÉ
dqn_213/dense_855/BiasAddBiasAdd"dqn_213/dense_855/MatMul:product:00dqn_213/dense_855/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_213/dense_855/BiasAddv
IdentityIdentity"dqn_213/dense_855/BiasAdd:output:0*
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
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:č
Ķ
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
[__call__"ļ
_tf_keras_modelÕ{"class_name": "DQN", "name": "dqn_213", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DQN"}}
š
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
]__call__"»
_tf_keras_layer”{"class_name": "Dense", "name": "dense_852", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_852", "trainable": true, "dtype": "float64", "units": 16, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_639", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
ņ

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*^&call_and_return_all_conditional_losses
___call__"½
_tf_keras_layer£{"class_name": "Dense", "name": "dense_853", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_853", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_640", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
ņ

activation

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"½
_tf_keras_layer£{"class_name": "Dense", "name": "dense_854", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_854", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_641", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
ņ

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_855", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_855", "trainable": true, "dtype": "float64", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
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
Ņ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Ć
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_639", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_639", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:(2dqn_213/dense_852/kernel
$:"2dqn_213/dense_852/bias
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
Ņ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
*g&call_and_return_all_conditional_losses
h__call__"Ć
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_640", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_640", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:( 2dqn_213/dense_853/kernel
$:" 2dqn_213/dense_853/bias
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
Ņ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
*i&call_and_return_all_conditional_losses
j__call__"Ć
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_641", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_641", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:(  2dqn_213/dense_854/kernel
$:" 2dqn_213/dense_854/bias
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
*:( 2dqn_213/dense_855/kernel
$:"2dqn_213/dense_855/bias
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
ą2Ż
"__inference__wrapped_model_4362470¶
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
2
D__inference_dqn_213_layer_call_and_return_conditional_losses_4362582Å
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
ö2ó
)__inference_dqn_213_layer_call_fn_4362604Å
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
š2ķ
F__inference_dense_852_layer_call_and_return_conditional_losses_4362638¢
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
Õ2Ņ
+__inference_dense_852_layer_call_fn_4362647¢
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
š2ķ
F__inference_dense_853_layer_call_and_return_conditional_losses_4362658¢
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
Õ2Ņ
+__inference_dense_853_layer_call_fn_4362667¢
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
š2ķ
F__inference_dense_854_layer_call_and_return_conditional_losses_4362678¢
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
Õ2Ņ
+__inference_dense_854_layer_call_fn_4362687¢
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
š2ķ
F__inference_dense_855_layer_call_and_return_conditional_losses_4362697¢
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
Õ2Ņ
+__inference_dense_855_layer_call_fn_4362706¢
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
4B2
%__inference_signature_wrapper_4362627input_1
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
 
"__inference__wrapped_model_4362470q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’¦
F__inference_dense_852_layer_call_and_return_conditional_losses_4362638\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_852_layer_call_fn_4362647O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¦
F__inference_dense_853_layer_call_and_return_conditional_losses_4362658\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’ 
 ~
+__inference_dense_853_layer_call_fn_4362667O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ ¦
F__inference_dense_854_layer_call_and_return_conditional_losses_4362678\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’ 
 ~
+__inference_dense_854_layer_call_fn_4362687O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’ ¦
F__inference_dense_855_layer_call_and_return_conditional_losses_4362697\ /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_855_layer_call_fn_4362706O /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’Å
D__inference_dqn_213_layer_call_and_return_conditional_losses_4362582} 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "?¢<
5Ŗ2
0
q_values$!

0/q_values’’’’’’’’’
 
)__inference_dqn_213_layer_call_fn_4362604q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’„
%__inference_signature_wrapper_4362627| ;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"3Ŗ0
.
q_values"
q_values’’’’’’’’’