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
dqn_139/dense_556/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namedqn_139/dense_556/kernel

,dqn_139/dense_556/kernel/Read/ReadVariableOpReadVariableOpdqn_139/dense_556/kernel*
_output_shapes

:*
dtype0

dqn_139/dense_556/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_139/dense_556/bias
}
*dqn_139/dense_556/bias/Read/ReadVariableOpReadVariableOpdqn_139/dense_556/bias*
_output_shapes
:*
dtype0

dqn_139/dense_557/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_139/dense_557/kernel

,dqn_139/dense_557/kernel/Read/ReadVariableOpReadVariableOpdqn_139/dense_557/kernel*
_output_shapes

: *
dtype0

dqn_139/dense_557/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_139/dense_557/bias
}
*dqn_139/dense_557/bias/Read/ReadVariableOpReadVariableOpdqn_139/dense_557/bias*
_output_shapes
: *
dtype0

dqn_139/dense_558/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_namedqn_139/dense_558/kernel

,dqn_139/dense_558/kernel/Read/ReadVariableOpReadVariableOpdqn_139/dense_558/kernel*
_output_shapes

:  *
dtype0

dqn_139/dense_558/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_139/dense_558/bias
}
*dqn_139/dense_558/bias/Read/ReadVariableOpReadVariableOpdqn_139/dense_558/bias*
_output_shapes
: *
dtype0

dqn_139/dense_559/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_139/dense_559/kernel

,dqn_139/dense_559/kernel/Read/ReadVariableOpReadVariableOpdqn_139/dense_559/kernel*
_output_shapes

: *
dtype0

dqn_139/dense_559/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_139/dense_559/bias
}
*dqn_139/dense_559/bias/Read/ReadVariableOpReadVariableOpdqn_139/dense_559/bias*
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
VARIABLE_VALUEdqn_139/dense_556/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_139/dense_556/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdqn_139/dense_557/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_139/dense_557/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdqn_139/dense_558/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEdqn_139/dense_558/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdqn_139/dense_559/kernel&dout/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEdqn_139/dense_559/bias$dout/bias/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn_139/dense_556/kerneldqn_139/dense_556/biasdqn_139/dense_557/kerneldqn_139/dense_557/biasdqn_139/dense_558/kerneldqn_139/dense_558/biasdqn_139/dense_559/kerneldqn_139/dense_559/bias*
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
%__inference_signature_wrapper_2100601
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,dqn_139/dense_556/kernel/Read/ReadVariableOp*dqn_139/dense_556/bias/Read/ReadVariableOp,dqn_139/dense_557/kernel/Read/ReadVariableOp*dqn_139/dense_557/bias/Read/ReadVariableOp,dqn_139/dense_558/kernel/Read/ReadVariableOp*dqn_139/dense_558/bias/Read/ReadVariableOp,dqn_139/dense_559/kernel/Read/ReadVariableOp*dqn_139/dense_559/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_2100727
ē
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_139/dense_556/kerneldqn_139/dense_556/biasdqn_139/dense_557/kerneldqn_139/dense_557/biasdqn_139/dense_558/kerneldqn_139/dense_558/biasdqn_139/dense_559/kerneldqn_139/dense_559/bias*
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
#__inference__traced_restore_2100761ń
	
®
F__inference_dense_556_layer_call_and_return_conditional_losses_2100459

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
leaky_re_lu_417/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_417/LeakyRelu{
IdentityIdentity'leaky_re_lu_417/LeakyRelu:activations:0*
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
®
F__inference_dense_557_layer_call_and_return_conditional_losses_2100486

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
leaky_re_lu_418/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_418/LeakyRelu{
IdentityIdentity'leaky_re_lu_418/LeakyRelu:activations:0*
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
Ļ
®
F__inference_dense_559_layer_call_and_return_conditional_losses_2100539

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
+__inference_dense_558_layer_call_fn_2100661

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
F__inference_dense_558_layer_call_and_return_conditional_losses_21005132
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
ź%
ķ
#__inference__traced_restore_2100761
file_prefix-
)assignvariableop_dqn_139_dense_556_kernel-
)assignvariableop_1_dqn_139_dense_556_bias/
+assignvariableop_2_dqn_139_dense_557_kernel-
)assignvariableop_3_dqn_139_dense_557_bias/
+assignvariableop_4_dqn_139_dense_558_kernel-
)assignvariableop_5_dqn_139_dense_558_bias/
+assignvariableop_6_dqn_139_dense_559_kernel-
)assignvariableop_7_dqn_139_dense_559_bias

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
AssignVariableOpAssignVariableOp)assignvariableop_dqn_139_dense_556_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_dqn_139_dense_556_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_dqn_139_dense_557_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp)assignvariableop_3_dqn_139_dense_557_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOp+assignvariableop_4_dqn_139_dense_558_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5®
AssignVariableOp_5AssignVariableOp)assignvariableop_5_dqn_139_dense_558_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOp+assignvariableop_6_dqn_139_dense_559_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7®
AssignVariableOp_7AssignVariableOp)assignvariableop_7_dqn_139_dense_559_biasIdentity_7:output:0"/device:CPU:0*
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
	
®
F__inference_dense_556_layer_call_and_return_conditional_losses_2100612

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
leaky_re_lu_417/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2
leaky_re_lu_417/LeakyRelu{
IdentityIdentity'leaky_re_lu_417/LeakyRelu:activations:0*
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

¬
D__inference_dqn_139_layer_call_and_return_conditional_losses_2100556
input_1
dense_556_2100470
dense_556_2100472
dense_557_2100497
dense_557_2100499
dense_558_2100524
dense_558_2100526
dense_559_2100550
dense_559_2100552
identity¢!dense_556/StatefulPartitionedCall¢!dense_557/StatefulPartitionedCall¢!dense_558/StatefulPartitionedCall¢!dense_559/StatefulPartitionedCall
!dense_556/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_556_2100470dense_556_2100472*
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
F__inference_dense_556_layer_call_and_return_conditional_losses_21004592#
!dense_556/StatefulPartitionedCallĄ
!dense_557/StatefulPartitionedCallStatefulPartitionedCall*dense_556/StatefulPartitionedCall:output:0dense_557_2100497dense_557_2100499*
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
F__inference_dense_557_layer_call_and_return_conditional_losses_21004862#
!dense_557/StatefulPartitionedCallĄ
!dense_558/StatefulPartitionedCallStatefulPartitionedCall*dense_557/StatefulPartitionedCall:output:0dense_558_2100524dense_558_2100526*
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
F__inference_dense_558_layer_call_and_return_conditional_losses_21005132#
!dense_558/StatefulPartitionedCallĄ
!dense_559/StatefulPartitionedCallStatefulPartitionedCall*dense_558/StatefulPartitionedCall:output:0dense_559_2100550dense_559_2100552*
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
F__inference_dense_559_layer_call_and_return_conditional_losses_21005392#
!dense_559/StatefulPartitionedCall
IdentityIdentity*dense_559/StatefulPartitionedCall:output:0"^dense_556/StatefulPartitionedCall"^dense_557/StatefulPartitionedCall"^dense_558/StatefulPartitionedCall"^dense_559/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2F
!dense_556/StatefulPartitionedCall!dense_556/StatefulPartitionedCall2F
!dense_557/StatefulPartitionedCall!dense_557/StatefulPartitionedCall2F
!dense_558/StatefulPartitionedCall!dense_558/StatefulPartitionedCall2F
!dense_559/StatefulPartitionedCall!dense_559/StatefulPartitionedCall:P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1

­
 __inference__traced_save_2100727
file_prefix7
3savev2_dqn_139_dense_556_kernel_read_readvariableop5
1savev2_dqn_139_dense_556_bias_read_readvariableop7
3savev2_dqn_139_dense_557_kernel_read_readvariableop5
1savev2_dqn_139_dense_557_bias_read_readvariableop7
3savev2_dqn_139_dense_558_kernel_read_readvariableop5
1savev2_dqn_139_dense_558_bias_read_readvariableop7
3savev2_dqn_139_dense_559_kernel_read_readvariableop5
1savev2_dqn_139_dense_559_bias_read_readvariableop
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
value3B1 B+_temp_37f81519e1d84e0aa8e08f176ac37cc0/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_dqn_139_dense_556_kernel_read_readvariableop1savev2_dqn_139_dense_556_bias_read_readvariableop3savev2_dqn_139_dense_557_kernel_read_readvariableop1savev2_dqn_139_dense_557_bias_read_readvariableop3savev2_dqn_139_dense_558_kernel_read_readvariableop1savev2_dqn_139_dense_558_bias_read_readvariableop3savev2_dqn_139_dense_559_kernel_read_readvariableop1savev2_dqn_139_dense_559_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
 
Ł
)__inference_dqn_139_layer_call_fn_2100578
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
D__inference_dqn_139_layer_call_and_return_conditional_losses_21005562
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
Ļ
®
F__inference_dense_559_layer_call_and_return_conditional_losses_2100671

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
Ą$
ö
"__inference__wrapped_model_2100444
input_14
0dqn_139_dense_556_matmul_readvariableop_resource5
1dqn_139_dense_556_biasadd_readvariableop_resource4
0dqn_139_dense_557_matmul_readvariableop_resource5
1dqn_139_dense_557_biasadd_readvariableop_resource4
0dqn_139_dense_558_matmul_readvariableop_resource5
1dqn_139_dense_558_biasadd_readvariableop_resource4
0dqn_139_dense_559_matmul_readvariableop_resource5
1dqn_139_dense_559_biasadd_readvariableop_resource
identityĆ
'dqn_139/dense_556/MatMul/ReadVariableOpReadVariableOp0dqn_139_dense_556_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'dqn_139/dense_556/MatMul/ReadVariableOpŖ
dqn_139/dense_556/MatMulMatMulinput_1/dqn_139/dense_556/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_139/dense_556/MatMulĀ
(dqn_139/dense_556/BiasAdd/ReadVariableOpReadVariableOp1dqn_139_dense_556_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(dqn_139/dense_556/BiasAdd/ReadVariableOpÉ
dqn_139/dense_556/BiasAddBiasAdd"dqn_139/dense_556/MatMul:product:00dqn_139/dense_556/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_139/dense_556/BiasAddĶ
+dqn_139/dense_556/leaky_re_lu_417/LeakyRelu	LeakyRelu"dqn_139/dense_556/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
alpha%>2-
+dqn_139/dense_556/leaky_re_lu_417/LeakyReluĆ
'dqn_139/dense_557/MatMul/ReadVariableOpReadVariableOp0dqn_139_dense_557_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'dqn_139/dense_557/MatMul/ReadVariableOpÜ
dqn_139/dense_557/MatMulMatMul9dqn_139/dense_556/leaky_re_lu_417/LeakyRelu:activations:0/dqn_139/dense_557/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_139/dense_557/MatMulĀ
(dqn_139/dense_557/BiasAdd/ReadVariableOpReadVariableOp1dqn_139_dense_557_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(dqn_139/dense_557/BiasAdd/ReadVariableOpÉ
dqn_139/dense_557/BiasAddBiasAdd"dqn_139/dense_557/MatMul:product:00dqn_139/dense_557/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_139/dense_557/BiasAddĶ
+dqn_139/dense_557/leaky_re_lu_418/LeakyRelu	LeakyRelu"dqn_139/dense_557/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2-
+dqn_139/dense_557/leaky_re_lu_418/LeakyReluĆ
'dqn_139/dense_558/MatMul/ReadVariableOpReadVariableOp0dqn_139_dense_558_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'dqn_139/dense_558/MatMul/ReadVariableOpÜ
dqn_139/dense_558/MatMulMatMul9dqn_139/dense_557/leaky_re_lu_418/LeakyRelu:activations:0/dqn_139/dense_558/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_139/dense_558/MatMulĀ
(dqn_139/dense_558/BiasAdd/ReadVariableOpReadVariableOp1dqn_139_dense_558_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(dqn_139/dense_558/BiasAdd/ReadVariableOpÉ
dqn_139/dense_558/BiasAddBiasAdd"dqn_139/dense_558/MatMul:product:00dqn_139/dense_558/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
dqn_139/dense_558/BiasAddĶ
+dqn_139/dense_558/leaky_re_lu_419/LeakyRelu	LeakyRelu"dqn_139/dense_558/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2-
+dqn_139/dense_558/leaky_re_lu_419/LeakyReluĆ
'dqn_139/dense_559/MatMul/ReadVariableOpReadVariableOp0dqn_139_dense_559_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'dqn_139/dense_559/MatMul/ReadVariableOpÜ
dqn_139/dense_559/MatMulMatMul9dqn_139/dense_558/leaky_re_lu_419/LeakyRelu:activations:0/dqn_139/dense_559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_139/dense_559/MatMulĀ
(dqn_139/dense_559/BiasAdd/ReadVariableOpReadVariableOp1dqn_139_dense_559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(dqn_139/dense_559/BiasAdd/ReadVariableOpÉ
dqn_139/dense_559/BiasAddBiasAdd"dqn_139/dense_559/MatMul:product:00dqn_139/dense_559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dqn_139/dense_559/BiasAddv
IdentityIdentity"dqn_139/dense_559/BiasAdd:output:0*
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
_user_specified_name	input_1
į

+__inference_dense_557_layer_call_fn_2100641

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
F__inference_dense_557_layer_call_and_return_conditional_losses_21004862
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
ś
Õ
%__inference_signature_wrapper_2100601
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
"__inference__wrapped_model_21004442
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
F__inference_dense_558_layer_call_and_return_conditional_losses_2100652

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
leaky_re_lu_419/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_419/LeakyRelu{
IdentityIdentity'leaky_re_lu_419/LeakyRelu:activations:0*
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
	
®
F__inference_dense_557_layer_call_and_return_conditional_losses_2100632

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
leaky_re_lu_418/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_418/LeakyRelu{
IdentityIdentity'leaky_re_lu_418/LeakyRelu:activations:0*
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
F__inference_dense_558_layer_call_and_return_conditional_losses_2100513

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
leaky_re_lu_419/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’ *
alpha%>2
leaky_re_lu_419/LeakyRelu{
IdentityIdentity'leaky_re_lu_419/LeakyRelu:activations:0*
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
į

+__inference_dense_556_layer_call_fn_2100621

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
F__inference_dense_556_layer_call_and_return_conditional_losses_21004592
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
+__inference_dense_559_layer_call_fn_2100680

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
F__inference_dense_559_layer_call_and_return_conditional_losses_21005392
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
 
_user_specified_nameinputs"øL
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
_tf_keras_modelÕ{"class_name": "DQN", "name": "dqn_139", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DQN"}}
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
_tf_keras_layer”{"class_name": "Dense", "name": "dense_556", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_556", "trainable": true, "dtype": "float64", "units": 16, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_417", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
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
_tf_keras_layer£{"class_name": "Dense", "name": "dense_557", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_557", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_418", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
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
_tf_keras_layer£{"class_name": "Dense", "name": "dense_558", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_558", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_419", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
ņ

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"Ķ
_tf_keras_layer³{"class_name": "Dense", "name": "dense_559", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_559", "trainable": true, "dtype": "float64", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
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
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_417", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_417", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:(2dqn_139/dense_556/kernel
$:"2dqn_139/dense_556/bias
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
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_418", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_418", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:( 2dqn_139/dense_557/kernel
$:" 2dqn_139/dense_557/bias
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
_tf_keras_layer©{"class_name": "LeakyReLU", "name": "leaky_re_lu_419", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_419", "trainable": true, "dtype": "float64", "alpha": 0.3}}
*:(  2dqn_139/dense_558/kernel
$:" 2dqn_139/dense_558/bias
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
*:( 2dqn_139/dense_559/kernel
$:"2dqn_139/dense_559/bias
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
"__inference__wrapped_model_2100444¶
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
D__inference_dqn_139_layer_call_and_return_conditional_losses_2100556Å
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
)__inference_dqn_139_layer_call_fn_2100578Å
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
F__inference_dense_556_layer_call_and_return_conditional_losses_2100612¢
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
+__inference_dense_556_layer_call_fn_2100621¢
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
F__inference_dense_557_layer_call_and_return_conditional_losses_2100632¢
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
+__inference_dense_557_layer_call_fn_2100641¢
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
F__inference_dense_558_layer_call_and_return_conditional_losses_2100652¢
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
+__inference_dense_558_layer_call_fn_2100661¢
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
F__inference_dense_559_layer_call_and_return_conditional_losses_2100671¢
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
+__inference_dense_559_layer_call_fn_2100680¢
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
%__inference_signature_wrapper_2100601input_1
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
"__inference__wrapped_model_2100444q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’¦
F__inference_dense_556_layer_call_and_return_conditional_losses_2100612\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_556_layer_call_fn_2100621O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¦
F__inference_dense_557_layer_call_and_return_conditional_losses_2100632\/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’ 
 ~
+__inference_dense_557_layer_call_fn_2100641O/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ ¦
F__inference_dense_558_layer_call_and_return_conditional_losses_2100652\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’ 
 ~
+__inference_dense_558_layer_call_fn_2100661O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’ ¦
F__inference_dense_559_layer_call_and_return_conditional_losses_2100671\ /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 ~
+__inference_dense_559_layer_call_fn_2100680O /¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’Å
D__inference_dqn_139_layer_call_and_return_conditional_losses_2100556} 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "?¢<
5Ŗ2
0
q_values$!

0/q_values’’’’’’’’’
 
)__inference_dqn_139_layer_call_fn_2100578q 0¢-
&¢#
!
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
q_values"
q_values’’’’’’’’’„
%__inference_signature_wrapper_2100601| ;¢8
¢ 
1Ŗ.
,
input_1!
input_1’’’’’’’’’"3Ŗ0
.
q_values"
q_values’’’’’’’’’