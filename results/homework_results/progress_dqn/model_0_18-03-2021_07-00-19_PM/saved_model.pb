││
┐Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02unknown8Ќд
є
dqn_5/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namedqn_5/dense_20/kernel

)dqn_5/dense_20/kernel/Read/ReadVariableOpReadVariableOpdqn_5/dense_20/kernel*
_output_shapes

:*
dtype0
~
dqn_5/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedqn_5/dense_20/bias
w
'dqn_5/dense_20/bias/Read/ReadVariableOpReadVariableOpdqn_5/dense_20/bias*
_output_shapes
:*
dtype0
є
dqn_5/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namedqn_5/dense_21/kernel

)dqn_5/dense_21/kernel/Read/ReadVariableOpReadVariableOpdqn_5/dense_21/kernel*
_output_shapes

: *
dtype0
~
dqn_5/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namedqn_5/dense_21/bias
w
'dqn_5/dense_21/bias/Read/ReadVariableOpReadVariableOpdqn_5/dense_21/bias*
_output_shapes
: *
dtype0
є
dqn_5/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_namedqn_5/dense_22/kernel

)dqn_5/dense_22/kernel/Read/ReadVariableOpReadVariableOpdqn_5/dense_22/kernel*
_output_shapes

:  *
dtype0
~
dqn_5/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namedqn_5/dense_22/bias
w
'dqn_5/dense_22/bias/Read/ReadVariableOpReadVariableOpdqn_5/dense_22/bias*
_output_shapes
: *
dtype0
є
dqn_5/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namedqn_5/dense_23/kernel

)dqn_5/dense_23/kernel/Read/ReadVariableOpReadVariableOpdqn_5/dense_23/kernel*
_output_shapes

: *
dtype0
~
dqn_5/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedqn_5/dense_23/bias
w
'dqn_5/dense_23/bias/Read/ReadVariableOpReadVariableOpdqn_5/dense_23/bias*
_output_shapes
:*
dtype0

NoOpNoOp
├
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*■
valueЗBы BЖ
ё
d1
d2
d3
dout
	variables
regularization_losses
trainable_variables
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
regularization_losses
trainable_variables
	keras_api
x

activation

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
x

activation

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
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
 
8
0
1
2
3
4
5
6
 7
Г
	variables
regularization_losses
%metrics
trainable_variables
&layer_regularization_losses

'layers
(non_trainable_variables
)layer_metrics
 
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
OM
VARIABLE_VALUEdqn_5/dense_20/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdqn_5/dense_20/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
	variables
.metrics
regularization_losses
trainable_variables
/layer_regularization_losses

0layers
1non_trainable_variables
2layer_metrics
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
OM
VARIABLE_VALUEdqn_5/dense_21/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdqn_5/dense_21/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
	variables
7metrics
regularization_losses
trainable_variables
8layer_regularization_losses

9layers
:non_trainable_variables
;layer_metrics
R
<	variables
=regularization_losses
>trainable_variables
?	keras_api
OM
VARIABLE_VALUEdqn_5/dense_22/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdqn_5/dense_22/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
	variables
@metrics
regularization_losses
trainable_variables
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
Dlayer_metrics
QO
VARIABLE_VALUEdqn_5/dense_23/kernel&dout/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdqn_5/dense_23/bias$dout/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
Г
!	variables
Emetrics
"regularization_losses
#trainable_variables
Flayer_regularization_losses

Glayers
Hnon_trainable_variables
Ilayer_metrics
 
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
Г
*	variables
Jmetrics
+regularization_losses
,trainable_variables
Klayer_regularization_losses

Llayers
Mnon_trainable_variables
Nlayer_metrics
 
 


0
 
 
 
 
 
Г
3	variables
Ometrics
4regularization_losses
5trainable_variables
Player_regularization_losses

Qlayers
Rnon_trainable_variables
Slayer_metrics
 
 

0
 
 
 
 
 
Г
<	variables
Tmetrics
=regularization_losses
>trainable_variables
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
Xlayer_metrics
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn_5/dense_20/kerneldqn_5/dense_20/biasdqn_5/dense_21/kerneldqn_5/dense_21/biasdqn_5/dense_22/kerneldqn_5/dense_22/biasdqn_5/dense_23/kerneldqn_5/dense_23/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_29529
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)dqn_5/dense_20/kernel/Read/ReadVariableOp'dqn_5/dense_20/bias/Read/ReadVariableOp)dqn_5/dense_21/kernel/Read/ReadVariableOp'dqn_5/dense_21/bias/Read/ReadVariableOp)dqn_5/dense_22/kernel/Read/ReadVariableOp'dqn_5/dense_22/bias/Read/ReadVariableOp)dqn_5/dense_23/kernel/Read/ReadVariableOp'dqn_5/dense_23/bias/Read/ReadVariableOpConst*
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
GPU 2J 8ѓ *'
f"R 
__inference__traced_save_29655
═
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_5/dense_20/kerneldqn_5/dense_20/biasdqn_5/dense_21/kerneldqn_5/dense_21/biasdqn_5/dense_22/kerneldqn_5/dense_22/biasdqn_5/dense_23/kerneldqn_5/dense_23/bias*
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_29689дь
┌
}
(__inference_dense_21_layer_call_fn_29569

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_294142
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
д
ї
@__inference_dqn_5_layer_call_and_return_conditional_losses_29484
input_1
dense_20_29398
dense_20_29400
dense_21_29425
dense_21_29427
dense_22_29452
dense_22_29454
dense_23_29478
dense_23_29480
identityѕб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallњ
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_29398dense_20_29400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_293872"
 dense_20/StatefulPartitionedCall┤
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_29425dense_21_29427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_21_layer_call_and_return_conditional_losses_294142"
 dense_21/StatefulPartitionedCall┤
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_29452dense_22_29454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_294412"
 dense_22/StatefulPartitionedCall┤
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_29478dense_23_29480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_294672"
 dense_23/StatefulPartitionedCallЅ
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
И%
М
!__inference__traced_restore_29689
file_prefix*
&assignvariableop_dqn_5_dense_20_kernel*
&assignvariableop_1_dqn_5_dense_20_bias,
(assignvariableop_2_dqn_5_dense_21_kernel*
&assignvariableop_3_dqn_5_dense_21_bias,
(assignvariableop_4_dqn_5_dense_22_kernel*
&assignvariableop_5_dqn_5_dense_22_bias,
(assignvariableop_6_dqn_5_dense_23_kernel*
&assignvariableop_7_dqn_5_dense_23_bias

identity_9ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7М
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*▀
valueНBм	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB&dout/kernel/.ATTRIBUTES/VARIABLE_VALUEB$dout/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesп
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

IdentityЦ
AssignVariableOpAssignVariableOp&assignvariableop_dqn_5_dense_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOp&assignvariableop_1_dqn_5_dense_20_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Г
AssignVariableOp_2AssignVariableOp(assignvariableop_2_dqn_5_dense_21_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ф
AssignVariableOp_3AssignVariableOp&assignvariableop_3_dqn_5_dense_21_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Г
AssignVariableOp_4AssignVariableOp(assignvariableop_4_dqn_5_dense_22_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ф
AssignVariableOp_5AssignVariableOp&assignvariableop_5_dqn_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Г
AssignVariableOp_6AssignVariableOp(assignvariableop_6_dqn_5_dense_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ф
AssignVariableOp_7AssignVariableOp&assignvariableop_7_dqn_5_dense_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpј

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8ђ

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
┌"
▄
 __inference__wrapped_model_29372
input_11
-dqn_5_dense_20_matmul_readvariableop_resource2
.dqn_5_dense_20_biasadd_readvariableop_resource1
-dqn_5_dense_21_matmul_readvariableop_resource2
.dqn_5_dense_21_biasadd_readvariableop_resource1
-dqn_5_dense_22_matmul_readvariableop_resource2
.dqn_5_dense_22_biasadd_readvariableop_resource1
-dqn_5_dense_23_matmul_readvariableop_resource2
.dqn_5_dense_23_biasadd_readvariableop_resource
identityѕ║
$dqn_5/dense_20/MatMul/ReadVariableOpReadVariableOp-dqn_5_dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$dqn_5/dense_20/MatMul/ReadVariableOpА
dqn_5/dense_20/MatMulMatMulinput_1,dqn_5/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dqn_5/dense_20/MatMul╣
%dqn_5/dense_20/BiasAdd/ReadVariableOpReadVariableOp.dqn_5_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%dqn_5/dense_20/BiasAdd/ReadVariableOpй
dqn_5/dense_20/BiasAddBiasAdddqn_5/dense_20/MatMul:product:0-dqn_5/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dqn_5/dense_20/BiasAdd┬
'dqn_5/dense_20/leaky_re_lu_15/LeakyRelu	LeakyReludqn_5/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         *
alpha%џЎЎ>2)
'dqn_5/dense_20/leaky_re_lu_15/LeakyRelu║
$dqn_5/dense_21/MatMul/ReadVariableOpReadVariableOp-dqn_5_dense_21_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$dqn_5/dense_21/MatMul/ReadVariableOp¤
dqn_5/dense_21/MatMulMatMul5dqn_5/dense_20/leaky_re_lu_15/LeakyRelu:activations:0,dqn_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dqn_5/dense_21/MatMul╣
%dqn_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp.dqn_5_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%dqn_5/dense_21/BiasAdd/ReadVariableOpй
dqn_5/dense_21/BiasAddBiasAdddqn_5/dense_21/MatMul:product:0-dqn_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dqn_5/dense_21/BiasAdd┬
'dqn_5/dense_21/leaky_re_lu_16/LeakyRelu	LeakyReludqn_5/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2)
'dqn_5/dense_21/leaky_re_lu_16/LeakyRelu║
$dqn_5/dense_22/MatMul/ReadVariableOpReadVariableOp-dqn_5_dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$dqn_5/dense_22/MatMul/ReadVariableOp¤
dqn_5/dense_22/MatMulMatMul5dqn_5/dense_21/leaky_re_lu_16/LeakyRelu:activations:0,dqn_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dqn_5/dense_22/MatMul╣
%dqn_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp.dqn_5_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%dqn_5/dense_22/BiasAdd/ReadVariableOpй
dqn_5/dense_22/BiasAddBiasAdddqn_5/dense_22/MatMul:product:0-dqn_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dqn_5/dense_22/BiasAdd┬
'dqn_5/dense_22/leaky_re_lu_17/LeakyRelu	LeakyReludqn_5/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2)
'dqn_5/dense_22/leaky_re_lu_17/LeakyRelu║
$dqn_5/dense_23/MatMul/ReadVariableOpReadVariableOp-dqn_5_dense_23_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$dqn_5/dense_23/MatMul/ReadVariableOp¤
dqn_5/dense_23/MatMulMatMul5dqn_5/dense_22/leaky_re_lu_17/LeakyRelu:activations:0,dqn_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dqn_5/dense_23/MatMul╣
%dqn_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp.dqn_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%dqn_5/dense_23/BiasAdd/ReadVariableOpй
dqn_5/dense_23/BiasAddBiasAdddqn_5/dense_23/MatMul:product:0-dqn_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dqn_5/dense_23/BiasAdds
IdentityIdentitydqn_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         :::::::::P L
'
_output_shapes
:         
!
_user_specified_name	input_1
ў
Н
%__inference_dqn_5_layer_call_fn_29506
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dqn_5_layer_call_and_return_conditional_losses_294842
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╠
Ф
C__inference_dense_23_layer_call_and_return_conditional_losses_29467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_22_layer_call_and_return_conditional_losses_29580

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddЋ
leaky_re_lu_17/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu_17/LeakyReluz
IdentityIdentity&leaky_re_lu_17/LeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_20_layer_call_and_return_conditional_losses_29387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
leaky_re_lu_15/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:         *
alpha%џЎЎ>2
leaky_re_lu_15/LeakyReluz
IdentityIdentity&leaky_re_lu_15/LeakyRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
}
(__inference_dense_20_layer_call_fn_29549

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_20_layer_call_and_return_conditional_losses_293872
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_22_layer_call_and_return_conditional_losses_29441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddЋ
leaky_re_lu_17/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu_17/LeakyReluz
IdentityIdentity&leaky_re_lu_17/LeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╬
Њ
__inference__traced_save_29655
file_prefix4
0savev2_dqn_5_dense_20_kernel_read_readvariableop2
.savev2_dqn_5_dense_20_bias_read_readvariableop4
0savev2_dqn_5_dense_21_kernel_read_readvariableop2
.savev2_dqn_5_dense_21_bias_read_readvariableop4
0savev2_dqn_5_dense_22_kernel_read_readvariableop2
.savev2_dqn_5_dense_22_bias_read_readvariableop4
0savev2_dqn_5_dense_23_kernel_read_readvariableop2
.savev2_dqn_5_dense_23_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_18b7090094b54cd6b9491079bca52fd7/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename═
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*▀
valueНBм	B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB&dout/kernel/.ATTRIBUTES/VARIABLE_VALUEB$dout/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesџ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices╩
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_dqn_5_dense_20_kernel_read_readvariableop.savev2_dqn_5_dense_20_bias_read_readvariableop0savev2_dqn_5_dense_21_kernel_read_readvariableop.savev2_dqn_5_dense_21_bias_read_readvariableop0savev2_dqn_5_dense_22_kernel_read_readvariableop.savev2_dqn_5_dense_22_bias_read_readvariableop0savev2_dqn_5_dense_23_kernel_read_readvariableop.savev2_dqn_5_dense_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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
Ш
М
#__inference_signature_wrapper_29529
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_293722
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
┌
}
(__inference_dense_22_layer_call_fn_29589

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_22_layer_call_and_return_conditional_losses_294412
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_21_layer_call_and_return_conditional_losses_29560

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddЋ
leaky_re_lu_16/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu_16/LeakyReluz
IdentityIdentity&leaky_re_lu_16/LeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_21_layer_call_and_return_conditional_losses_29414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddЋ
leaky_re_lu_16/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:          *
alpha%џЎЎ>2
leaky_re_lu_16/LeakyReluz
IdentityIdentity&leaky_re_lu_16/LeakyRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
}
(__inference_dense_23_layer_call_fn_29608

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_294672
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Щ
Ф
C__inference_dense_20_layer_call_and_return_conditional_losses_29540

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
leaky_re_lu_15/LeakyRelu	LeakyReluBiasAdd:output:0*
T0*'
_output_shapes
:         *
alpha%џЎЎ>2
leaky_re_lu_15/LeakyReluz
IdentityIdentity&leaky_re_lu_15/LeakyRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╠
Ф
C__inference_dense_23_layer_call_and_return_conditional_losses_29599

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultЌ
;
input_10
serving_default_input_1:0         <
q_values0
StatefulPartitionedCall:0         tensorflow/serving/predict:шЁ
╦
d1
d2
d3
dout
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
Y__call__
Z_default_save_signature
*[&call_and_return_all_conditional_losses"ь
_tf_keras_modelМ{"class_name": "DQN", "name": "dqn_5", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DQN"}}
ь


activation

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
\__call__
*]&call_and_return_all_conditional_losses"И
_tf_keras_layerъ{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float64", "units": 16, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_15", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
№

activation

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_16", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
№

activation

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float64", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_17", "trainable": true, "dtype": "float64", "alpha": 0.3}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
­

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
b__call__
*c&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float64", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
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
╩
	variables
regularization_losses
%metrics
trainable_variables
&layer_regularization_losses

'layers
(non_trainable_variables
)layer_metrics
Y__call__
Z_default_save_signature
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
,
dserving_default"
signature_map
л
*	variables
+regularization_losses
,trainable_variables
-	keras_api
e__call__
*f&call_and_return_all_conditional_losses"┴
_tf_keras_layerД{"class_name": "LeakyReLU", "name": "leaky_re_lu_15", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_15", "trainable": true, "dtype": "float64", "alpha": 0.3}}
':%2dqn_5/dense_20/kernel
!:2dqn_5/dense_20/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
	variables
.metrics
regularization_losses
trainable_variables
/layer_regularization_losses

0layers
1non_trainable_variables
2layer_metrics
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
л
3	variables
4regularization_losses
5trainable_variables
6	keras_api
g__call__
*h&call_and_return_all_conditional_losses"┴
_tf_keras_layerД{"class_name": "LeakyReLU", "name": "leaky_re_lu_16", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_16", "trainable": true, "dtype": "float64", "alpha": 0.3}}
':% 2dqn_5/dense_21/kernel
!: 2dqn_5/dense_21/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
	variables
7metrics
regularization_losses
trainable_variables
8layer_regularization_losses

9layers
:non_trainable_variables
;layer_metrics
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
л
<	variables
=regularization_losses
>trainable_variables
?	keras_api
i__call__
*j&call_and_return_all_conditional_losses"┴
_tf_keras_layerД{"class_name": "LeakyReLU", "name": "leaky_re_lu_17", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_17", "trainable": true, "dtype": "float64", "alpha": 0.3}}
':%  2dqn_5/dense_22/kernel
!: 2dqn_5/dense_22/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
	variables
@metrics
regularization_losses
trainable_variables
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
Dlayer_metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
':% 2dqn_5/dense_23/kernel
!:2dqn_5/dense_23/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
Г
!	variables
Emetrics
"regularization_losses
#trainable_variables
Flayer_regularization_losses

Glayers
Hnon_trainable_variables
Ilayer_metrics
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
Г
*	variables
Jmetrics
+regularization_losses
,trainable_variables
Klayer_regularization_losses

Llayers
Mnon_trainable_variables
Nlayer_metrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
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
Г
3	variables
Ometrics
4regularization_losses
5trainable_variables
Player_regularization_losses

Qlayers
Rnon_trainable_variables
Slayer_metrics
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
Г
<	variables
Tmetrics
=regularization_losses
>trainable_variables
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
Xlayer_metrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ы2№
%__inference_dqn_5_layer_call_fn_29506┼
ў▓ћ
FullArgSpec
argsџ
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
я2█
 __inference__wrapped_model_29372Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
Ї2і
@__inference_dqn_5_layer_call_and_return_conditional_losses_29484┼
ў▓ћ
FullArgSpec
argsџ
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         
м2¤
(__inference_dense_20_layer_call_fn_29549б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_20_layer_call_and_return_conditional_losses_29540б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_21_layer_call_fn_29569б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_21_layer_call_and_return_conditional_losses_29560б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_22_layer_call_fn_29589б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_22_layer_call_and_return_conditional_losses_29580б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_23_layer_call_fn_29608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_23_layer_call_and_return_conditional_losses_29599б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
2B0
#__inference_signature_wrapper_29529input_1
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ћ
 __inference__wrapped_model_29372q 0б-
&б#
!і
input_1         
ф "3ф0
.
q_values"і
q_values         Б
C__inference_dense_20_layer_call_and_return_conditional_losses_29540\/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ {
(__inference_dense_20_layer_call_fn_29549O/б,
%б"
 і
inputs         
ф "і         Б
C__inference_dense_21_layer_call_and_return_conditional_losses_29560\/б,
%б"
 і
inputs         
ф "%б"
і
0          
џ {
(__inference_dense_21_layer_call_fn_29569O/б,
%б"
 і
inputs         
ф "і          Б
C__inference_dense_22_layer_call_and_return_conditional_losses_29580\/б,
%б"
 і
inputs          
ф "%б"
і
0          
џ {
(__inference_dense_22_layer_call_fn_29589O/б,
%б"
 і
inputs          
ф "і          Б
C__inference_dense_23_layer_call_and_return_conditional_losses_29599\ /б,
%б"
 і
inputs          
ф "%б"
і
0         
џ {
(__inference_dense_23_layer_call_fn_29608O /б,
%б"
 і
inputs          
ф "і         ┴
@__inference_dqn_5_layer_call_and_return_conditional_losses_29484} 0б-
&б#
!і
input_1         
ф "?б<
5ф2
0
q_values$і!

0/q_values         
џ џ
%__inference_dqn_5_layer_call_fn_29506q 0б-
&б#
!і
input_1         
ф "3ф0
.
q_values"і
q_values         Б
#__inference_signature_wrapper_29529| ;б8
б 
1ф.
,
input_1!і
input_1         "3ф0
.
q_values"і
q_values         