Ŝ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
p
	CS/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	CS/kernel
i
CS/kernel/Read/ReadVariableOpReadVariableOp	CS/kernel* 
_output_shapes
:
??*
dtype0
f
CS/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	CS/bias
_
CS/bias/Read/ReadVariableOpReadVariableOpCS/bias*
_output_shapes
:*
dtype0
p
	PH/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	PH/kernel
i
PH/kernel/Read/ReadVariableOpReadVariableOp	PH/kernel* 
_output_shapes
:
??*
dtype0
f
PH/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	PH/bias
_
PH/bias/Read/ReadVariableOpReadVariableOpPH/bias*
_output_shapes
:*
dtype0
t
MATH/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameMATH/kernel
m
MATH/kernel/Read/ReadVariableOpReadVariableOpMATH/kernel* 
_output_shapes
:
??*
dtype0
j
	MATH/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	MATH/bias
c
MATH/bias/Read/ReadVariableOpReadVariableOp	MATH/bias*
_output_shapes
:*
dtype0
t
STAT/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameSTAT/kernel
m
STAT/kernel/Read/ReadVariableOpReadVariableOpSTAT/kernel* 
_output_shapes
:
??*
dtype0
j
	STAT/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	STAT/bias
c
STAT/bias/Read/ReadVariableOpReadVariableOp	STAT/bias*
_output_shapes
:*
dtype0
p
	QB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	QB/kernel
i
QB/kernel/Read/ReadVariableOpReadVariableOp	QB/kernel* 
_output_shapes
:
??*
dtype0
f
QB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	QB/bias
_
QB/bias/Read/ReadVariableOpReadVariableOpQB/bias*
_output_shapes
:*
dtype0
p
	QF/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	QF/kernel
i
QF/kernel/Read/ReadVariableOpReadVariableOp	QF/kernel* 
_output_shapes
:
??*
dtype0
f
QF/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	QF/bias
_
QF/bias/Read/ReadVariableOpReadVariableOpQF/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
~
Adam/CS/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/CS/kernel/m
w
$Adam/CS/kernel/m/Read/ReadVariableOpReadVariableOpAdam/CS/kernel/m* 
_output_shapes
:
??*
dtype0
t
Adam/CS/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/CS/bias/m
m
"Adam/CS/bias/m/Read/ReadVariableOpReadVariableOpAdam/CS/bias/m*
_output_shapes
:*
dtype0
~
Adam/PH/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/PH/kernel/m
w
$Adam/PH/kernel/m/Read/ReadVariableOpReadVariableOpAdam/PH/kernel/m* 
_output_shapes
:
??*
dtype0
t
Adam/PH/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/PH/bias/m
m
"Adam/PH/bias/m/Read/ReadVariableOpReadVariableOpAdam/PH/bias/m*
_output_shapes
:*
dtype0
?
Adam/MATH/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/MATH/kernel/m
{
&Adam/MATH/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MATH/kernel/m* 
_output_shapes
:
??*
dtype0
x
Adam/MATH/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/MATH/bias/m
q
$Adam/MATH/bias/m/Read/ReadVariableOpReadVariableOpAdam/MATH/bias/m*
_output_shapes
:*
dtype0
?
Adam/STAT/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/STAT/kernel/m
{
&Adam/STAT/kernel/m/Read/ReadVariableOpReadVariableOpAdam/STAT/kernel/m* 
_output_shapes
:
??*
dtype0
x
Adam/STAT/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/STAT/bias/m
q
$Adam/STAT/bias/m/Read/ReadVariableOpReadVariableOpAdam/STAT/bias/m*
_output_shapes
:*
dtype0
~
Adam/QB/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/QB/kernel/m
w
$Adam/QB/kernel/m/Read/ReadVariableOpReadVariableOpAdam/QB/kernel/m* 
_output_shapes
:
??*
dtype0
t
Adam/QB/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/QB/bias/m
m
"Adam/QB/bias/m/Read/ReadVariableOpReadVariableOpAdam/QB/bias/m*
_output_shapes
:*
dtype0
~
Adam/QF/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/QF/kernel/m
w
$Adam/QF/kernel/m/Read/ReadVariableOpReadVariableOpAdam/QF/kernel/m* 
_output_shapes
:
??*
dtype0
t
Adam/QF/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/QF/bias/m
m
"Adam/QF/bias/m/Read/ReadVariableOpReadVariableOpAdam/QF/bias/m*
_output_shapes
:*
dtype0
~
Adam/CS/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/CS/kernel/v
w
$Adam/CS/kernel/v/Read/ReadVariableOpReadVariableOpAdam/CS/kernel/v* 
_output_shapes
:
??*
dtype0
t
Adam/CS/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/CS/bias/v
m
"Adam/CS/bias/v/Read/ReadVariableOpReadVariableOpAdam/CS/bias/v*
_output_shapes
:*
dtype0
~
Adam/PH/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/PH/kernel/v
w
$Adam/PH/kernel/v/Read/ReadVariableOpReadVariableOpAdam/PH/kernel/v* 
_output_shapes
:
??*
dtype0
t
Adam/PH/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/PH/bias/v
m
"Adam/PH/bias/v/Read/ReadVariableOpReadVariableOpAdam/PH/bias/v*
_output_shapes
:*
dtype0
?
Adam/MATH/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/MATH/kernel/v
{
&Adam/MATH/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MATH/kernel/v* 
_output_shapes
:
??*
dtype0
x
Adam/MATH/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/MATH/bias/v
q
$Adam/MATH/bias/v/Read/ReadVariableOpReadVariableOpAdam/MATH/bias/v*
_output_shapes
:*
dtype0
?
Adam/STAT/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/STAT/kernel/v
{
&Adam/STAT/kernel/v/Read/ReadVariableOpReadVariableOpAdam/STAT/kernel/v* 
_output_shapes
:
??*
dtype0
x
Adam/STAT/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/STAT/bias/v
q
$Adam/STAT/bias/v/Read/ReadVariableOpReadVariableOpAdam/STAT/bias/v*
_output_shapes
:*
dtype0
~
Adam/QB/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/QB/kernel/v
w
$Adam/QB/kernel/v/Read/ReadVariableOpReadVariableOpAdam/QB/kernel/v* 
_output_shapes
:
??*
dtype0
t
Adam/QB/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/QB/bias/v
m
"Adam/QB/bias/v/Read/ReadVariableOpReadVariableOpAdam/QB/bias/v*
_output_shapes
:*
dtype0
~
Adam/QF/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_nameAdam/QF/kernel/v
w
$Adam/QF/kernel/v/Read/ReadVariableOpReadVariableOpAdam/QF/kernel/v* 
_output_shapes
:
??*
dtype0
t
Adam/QF/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/QF/bias/v
m
"Adam/QF/bias/v/Read/ReadVariableOpReadVariableOpAdam/QF/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?X
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?X
value?XB?W B?W
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratem?m?m?m? m?!m?&m?'m?,m?-m?2m?3m?v?v?v?v? v?!v?&v?'v?,v?-v?2v?3v?
 
V
0
1
2
3
 4
!5
&6
'7
,8
-9
210
311
V
0
1
2
3
 4
!5
&6
'7
,8
-9
210
311
?
=non_trainable_variables

>layers
regularization_losses
trainable_variables
?layer_metrics
@layer_regularization_losses
	variables
Ametrics
 
 
 
 
?
Bnon_trainable_variables

Clayers
regularization_losses
trainable_variables
Dlayer_metrics
Elayer_regularization_losses
	variables
Fmetrics
US
VARIABLE_VALUE	CS/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUECS/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Gnon_trainable_variables

Hlayers
regularization_losses
trainable_variables
Ilayer_metrics
Jlayer_regularization_losses
	variables
Kmetrics
US
VARIABLE_VALUE	PH/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEPH/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Lnon_trainable_variables

Mlayers
regularization_losses
trainable_variables
Nlayer_metrics
Olayer_regularization_losses
	variables
Pmetrics
WU
VARIABLE_VALUEMATH/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	MATH/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
Qnon_trainable_variables

Rlayers
"regularization_losses
#trainable_variables
Slayer_metrics
Tlayer_regularization_losses
$	variables
Umetrics
WU
VARIABLE_VALUESTAT/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	STAT/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
?
Vnon_trainable_variables

Wlayers
(regularization_losses
)trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
*	variables
Zmetrics
US
VARIABLE_VALUE	QB/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEQB/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
[non_trainable_variables

\layers
.regularization_losses
/trainable_variables
]layer_metrics
^layer_regularization_losses
0	variables
_metrics
US
VARIABLE_VALUE	QF/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEQF/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
?
`non_trainable_variables

alayers
4regularization_losses
5trainable_variables
blayer_metrics
clayer_regularization_losses
6	variables
dmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8
 
 
^
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
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
4
	rtotal
	scount
t	variables
u	keras_api
4
	vtotal
	wcount
x	variables
y	keras_api
4
	ztotal
	{count
|	variables
}	keras_api
6
	~total
	count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

t	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

v0
w1

x	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

|	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

~0
1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
xv
VARIABLE_VALUEAdam/CS/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/CS/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/PH/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/PH/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/MATH/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/MATH/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/STAT/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/STAT/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/QB/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/QB/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/QF/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/QF/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/CS/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/CS/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/PH/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/PH/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/MATH/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/MATH/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/STAT/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/STAT/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/QB/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/QB/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/QF/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/QF/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_abstractPlaceholder*)
_output_shapes
:???????????*
dtype0*
shape:???????????
|
serving_default_titlePlaceholder*)
_output_shapes
:???????????*
dtype0*
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_abstractserving_default_title	QF/kernelQF/bias	QB/kernelQB/biasSTAT/kernel	STAT/biasMATH/kernel	MATH/bias	PH/kernelPH/bias	CS/kernelCS/bias*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_109506
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameCS/kernel/Read/ReadVariableOpCS/bias/Read/ReadVariableOpPH/kernel/Read/ReadVariableOpPH/bias/Read/ReadVariableOpMATH/kernel/Read/ReadVariableOpMATH/bias/Read/ReadVariableOpSTAT/kernel/Read/ReadVariableOpSTAT/bias/Read/ReadVariableOpQB/kernel/Read/ReadVariableOpQB/bias/Read/ReadVariableOpQF/kernel/Read/ReadVariableOpQF/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOp$Adam/CS/kernel/m/Read/ReadVariableOp"Adam/CS/bias/m/Read/ReadVariableOp$Adam/PH/kernel/m/Read/ReadVariableOp"Adam/PH/bias/m/Read/ReadVariableOp&Adam/MATH/kernel/m/Read/ReadVariableOp$Adam/MATH/bias/m/Read/ReadVariableOp&Adam/STAT/kernel/m/Read/ReadVariableOp$Adam/STAT/bias/m/Read/ReadVariableOp$Adam/QB/kernel/m/Read/ReadVariableOp"Adam/QB/bias/m/Read/ReadVariableOp$Adam/QF/kernel/m/Read/ReadVariableOp"Adam/QF/bias/m/Read/ReadVariableOp$Adam/CS/kernel/v/Read/ReadVariableOp"Adam/CS/bias/v/Read/ReadVariableOp$Adam/PH/kernel/v/Read/ReadVariableOp"Adam/PH/bias/v/Read/ReadVariableOp&Adam/MATH/kernel/v/Read/ReadVariableOp$Adam/MATH/bias/v/Read/ReadVariableOp&Adam/STAT/kernel/v/Read/ReadVariableOp$Adam/STAT/bias/v/Read/ReadVariableOp$Adam/QB/kernel/v/Read/ReadVariableOp"Adam/QB/bias/v/Read/ReadVariableOp$Adam/QF/kernel/v/Read/ReadVariableOp"Adam/QF/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_110501
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	CS/kernelCS/bias	PH/kernelPH/biasMATH/kernel	MATH/biasSTAT/kernel	STAT/bias	QB/kernelQB/bias	QF/kernelQF/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10total_11count_11total_12count_12Adam/CS/kernel/mAdam/CS/bias/mAdam/PH/kernel/mAdam/PH/bias/mAdam/MATH/kernel/mAdam/MATH/bias/mAdam/STAT/kernel/mAdam/STAT/bias/mAdam/QB/kernel/mAdam/QB/bias/mAdam/QF/kernel/mAdam/QF/bias/mAdam/CS/kernel/vAdam/CS/bias/vAdam/PH/kernel/vAdam/PH/bias/vAdam/MATH/kernel/vAdam/MATH/bias/vAdam/STAT/kernel/vAdam/STAT/bias/vAdam/QB/kernel/vAdam/QB/bias/vAdam/QF/kernel/vAdam/QF/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_110712??
??
?
A__inference_model_layer_call_and_return_conditional_losses_109377

inputs
inputs_1
	qf_109251
	qf_109253
	qb_109264
	qb_109266
stat_109277
stat_109279
math_109290
math_109292
	ph_109303
	ph_109305
	cs_109316
	cs_109318
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/StatefulPartitionedCall?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/StatefulPartitionedCall?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/StatefulPartitionedCall?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/StatefulPartitionedCall?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/StatefulPartitionedCall?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/StatefulPartitionedCall?-STAT/kernel/Regularizer/Square/ReadVariableOp?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1085592
concatenate/PartitionedCall?
QF/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qf_109251	qf_109253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
QF/StatefulPartitionedCall?
&QF/ActivityRegularizer/PartitionedCallPartitionedCall#QF/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QF_activity_regularizer_1085472(
&QF/ActivityRegularizer/PartitionedCall?
QF/ActivityRegularizer/ShapeShape#QF/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDiv/QF/ActivityRegularizer/PartitionedCall:output:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qb_109264	qb_109266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
QB/StatefulPartitionedCall?
&QB/ActivityRegularizer/PartitionedCallPartitionedCall#QB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QB_activity_regularizer_1085342(
&QB/ActivityRegularizer/PartitionedCall?
QB/ActivityRegularizer/ShapeShape#QB/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDiv/QB/ActivityRegularizer/PartitionedCall:output:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0stat_109277stat_109279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
STAT/StatefulPartitionedCall?
(STAT/ActivityRegularizer/PartitionedCallPartitionedCall%STAT/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_STAT_activity_regularizer_1085212*
(STAT/ActivityRegularizer/PartitionedCall?
STAT/ActivityRegularizer/ShapeShape%STAT/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv1STAT/ActivityRegularizer/PartitionedCall:output:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0math_109290math_109292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
MATH/StatefulPartitionedCall?
(MATH/ActivityRegularizer/PartitionedCallPartitionedCall%MATH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_MATH_activity_regularizer_1085082*
(MATH/ActivityRegularizer/PartitionedCall?
MATH/ActivityRegularizer/ShapeShape%MATH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv1MATH/ActivityRegularizer/PartitionedCall:output:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	ph_109303	ph_109305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
PH/StatefulPartitionedCall?
&PH/ActivityRegularizer/PartitionedCallPartitionedCall#PH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_PH_activity_regularizer_1084952(
&PH/ActivityRegularizer/PartitionedCall?
PH/ActivityRegularizer/ShapeShape#PH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDiv/PH/ActivityRegularizer/PartitionedCall:output:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	cs_109316	cs_109318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
CS/StatefulPartitionedCall?
&CS/ActivityRegularizer/PartitionedCallPartitionedCall#CS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_CS_activity_regularizer_1084822(
&CS/ActivityRegularizer/PartitionedCall?
CS/ActivityRegularizer/ShapeShape#CS/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDiv/CS/ActivityRegularizer/PartitionedCall:output:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	cs_109316* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	ph_109303* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmath_109290* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstat_109277* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qb_109264* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qf_109251* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentity#CS/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#PH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%MATH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity%STAT/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#QB/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#QF/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::28
CS/StatefulPartitionedCallCS/StatefulPartitionedCall2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2<
MATH/StatefulPartitionedCallMATH/StatefulPartitionedCall2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp28
PH/StatefulPartitionedCallPH/StatefulPartitionedCall2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp28
QB/StatefulPartitionedCallQB/StatefulPartitionedCall2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp28
QF/StatefulPartitionedCallQF/StatefulPartitionedCall2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2<
STAT/StatefulPartitionedCallSTAT/StatefulPartitionedCall2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:QM
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>__inference_QF_layer_call_and_return_conditional_losses_108585

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+QF/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>__inference_QB_layer_call_and_return_conditional_losses_108638

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+QB/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?v
?
__inference__traced_save_110501
file_prefix(
$savev2_cs_kernel_read_readvariableop&
"savev2_cs_bias_read_readvariableop(
$savev2_ph_kernel_read_readvariableop&
"savev2_ph_bias_read_readvariableop*
&savev2_math_kernel_read_readvariableop(
$savev2_math_bias_read_readvariableop*
&savev2_stat_kernel_read_readvariableop(
$savev2_stat_bias_read_readvariableop(
$savev2_qb_kernel_read_readvariableop&
"savev2_qb_bias_read_readvariableop(
$savev2_qf_kernel_read_readvariableop&
"savev2_qf_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableop'
#savev2_total_11_read_readvariableop'
#savev2_count_11_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop/
+savev2_adam_cs_kernel_m_read_readvariableop-
)savev2_adam_cs_bias_m_read_readvariableop/
+savev2_adam_ph_kernel_m_read_readvariableop-
)savev2_adam_ph_bias_m_read_readvariableop1
-savev2_adam_math_kernel_m_read_readvariableop/
+savev2_adam_math_bias_m_read_readvariableop1
-savev2_adam_stat_kernel_m_read_readvariableop/
+savev2_adam_stat_bias_m_read_readvariableop/
+savev2_adam_qb_kernel_m_read_readvariableop-
)savev2_adam_qb_bias_m_read_readvariableop/
+savev2_adam_qf_kernel_m_read_readvariableop-
)savev2_adam_qf_bias_m_read_readvariableop/
+savev2_adam_cs_kernel_v_read_readvariableop-
)savev2_adam_cs_bias_v_read_readvariableop/
+savev2_adam_ph_kernel_v_read_readvariableop-
)savev2_adam_ph_bias_v_read_readvariableop1
-savev2_adam_math_kernel_v_read_readvariableop/
+savev2_adam_math_bias_v_read_readvariableop1
-savev2_adam_stat_kernel_v_read_readvariableop/
+savev2_adam_stat_bias_v_read_readvariableop/
+savev2_adam_qb_kernel_v_read_readvariableop-
)savev2_adam_qb_bias_v_read_readvariableop/
+savev2_adam_qf_kernel_v_read_readvariableop-
)savev2_adam_qf_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?!
value?!B?!DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_cs_kernel_read_readvariableop"savev2_cs_bias_read_readvariableop$savev2_ph_kernel_read_readvariableop"savev2_ph_bias_read_readvariableop&savev2_math_kernel_read_readvariableop$savev2_math_bias_read_readvariableop&savev2_stat_kernel_read_readvariableop$savev2_stat_bias_read_readvariableop$savev2_qb_kernel_read_readvariableop"savev2_qb_bias_read_readvariableop$savev2_qf_kernel_read_readvariableop"savev2_qf_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableop#savev2_total_11_read_readvariableop#savev2_count_11_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop+savev2_adam_cs_kernel_m_read_readvariableop)savev2_adam_cs_bias_m_read_readvariableop+savev2_adam_ph_kernel_m_read_readvariableop)savev2_adam_ph_bias_m_read_readvariableop-savev2_adam_math_kernel_m_read_readvariableop+savev2_adam_math_bias_m_read_readvariableop-savev2_adam_stat_kernel_m_read_readvariableop+savev2_adam_stat_bias_m_read_readvariableop+savev2_adam_qb_kernel_m_read_readvariableop)savev2_adam_qb_bias_m_read_readvariableop+savev2_adam_qf_kernel_m_read_readvariableop)savev2_adam_qf_bias_m_read_readvariableop+savev2_adam_cs_kernel_v_read_readvariableop)savev2_adam_cs_bias_v_read_readvariableop+savev2_adam_ph_kernel_v_read_readvariableop)savev2_adam_ph_bias_v_read_readvariableop-savev2_adam_math_kernel_v_read_readvariableop+savev2_adam_math_bias_v_read_readvariableop-savev2_adam_stat_kernel_v_read_readvariableop+savev2_adam_stat_bias_v_read_readvariableop+savev2_adam_qb_kernel_v_read_readvariableop)savev2_adam_qb_bias_v_read_readvariableop+savev2_adam_qf_kernel_v_read_readvariableop)savev2_adam_qf_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??::
??::
??::
??::
??::
??:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
??::
??::
??::
??::
??::
??::
??::
??::
??::
??::
??::
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??: 

_output_shapes
::&"
 
_output_shapes
:
??: 

_output_shapes
::&"
 
_output_shapes
:
??: 

_output_shapes
::&"
 
_output_shapes
:
??: 

_output_shapes
::&	"
 
_output_shapes
:
??: 


_output_shapes
::&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :&,"
 
_output_shapes
:
??: -

_output_shapes
::&."
 
_output_shapes
:
??: /

_output_shapes
::&0"
 
_output_shapes
:
??: 1

_output_shapes
::&2"
 
_output_shapes
:
??: 3

_output_shapes
::&4"
 
_output_shapes
:
??: 5

_output_shapes
::&6"
 
_output_shapes
:
??: 7

_output_shapes
::&8"
 
_output_shapes
:
??: 9

_output_shapes
::&:"
 
_output_shapes
:
??: ;

_output_shapes
::&<"
 
_output_shapes
:
??: =

_output_shapes
::&>"
 
_output_shapes
:
??: ?

_output_shapes
::&@"
 
_output_shapes
:
??: A

_output_shapes
::&B"
 
_output_shapes
:
??: C

_output_shapes
::D

_output_shapes
: 
?
D
*__inference_PH_activity_regularizer_108495
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
??
?
!__inference__wrapped_model_108469	
title
abstract+
'model_qf_matmul_readvariableop_resource,
(model_qf_biasadd_readvariableop_resource+
'model_qb_matmul_readvariableop_resource,
(model_qb_biasadd_readvariableop_resource-
)model_stat_matmul_readvariableop_resource.
*model_stat_biasadd_readvariableop_resource-
)model_math_matmul_readvariableop_resource.
*model_math_biasadd_readvariableop_resource+
'model_ph_matmul_readvariableop_resource,
(model_ph_biasadd_readvariableop_resource+
'model_cs_matmul_readvariableop_resource,
(model_cs_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5??model/CS/BiasAdd/ReadVariableOp?model/CS/MatMul/ReadVariableOp?!model/MATH/BiasAdd/ReadVariableOp? model/MATH/MatMul/ReadVariableOp?model/PH/BiasAdd/ReadVariableOp?model/PH/MatMul/ReadVariableOp?model/QB/BiasAdd/ReadVariableOp?model/QB/MatMul/ReadVariableOp?model/QF/BiasAdd/ReadVariableOp?model/QF/MatMul/ReadVariableOp?!model/STAT/BiasAdd/ReadVariableOp? model/STAT/MatMul/ReadVariableOp?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2titleabstract&model/concatenate/concat/axis:output:0*
N*
T0*)
_output_shapes
:???????????2
model/concatenate/concat?
model/QF/MatMul/ReadVariableOpReadVariableOp'model_qf_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/QF/MatMul/ReadVariableOp?
model/QF/MatMulMatMul!model/concatenate/concat:output:0&model/QF/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/QF/MatMul?
model/QF/BiasAdd/ReadVariableOpReadVariableOp(model_qf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
model/QF/BiasAdd/ReadVariableOp?
model/QF/BiasAddBiasAddmodel/QF/MatMul:product:0'model/QF/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/QF/BiasAdd|
model/QF/SigmoidSigmoidmodel/QF/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/QF/Sigmoid?
#model/QF/ActivityRegularizer/SquareSquaremodel/QF/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#model/QF/ActivityRegularizer/Square?
"model/QF/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model/QF/ActivityRegularizer/Const?
 model/QF/ActivityRegularizer/SumSum'model/QF/ActivityRegularizer/Square:y:0+model/QF/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 model/QF/ActivityRegularizer/Sum?
"model/QF/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"model/QF/ActivityRegularizer/mul/x?
 model/QF/ActivityRegularizer/mulMul+model/QF/ActivityRegularizer/mul/x:output:0)model/QF/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 model/QF/ActivityRegularizer/mul?
"model/QF/ActivityRegularizer/ShapeShapemodel/QF/Sigmoid:y:0*
T0*
_output_shapes
:2$
"model/QF/ActivityRegularizer/Shape?
0model/QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model/QF/ActivityRegularizer/strided_slice/stack?
2model/QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model/QF/ActivityRegularizer/strided_slice/stack_1?
2model/QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model/QF/ActivityRegularizer/strided_slice/stack_2?
*model/QF/ActivityRegularizer/strided_sliceStridedSlice+model/QF/ActivityRegularizer/Shape:output:09model/QF/ActivityRegularizer/strided_slice/stack:output:0;model/QF/ActivityRegularizer/strided_slice/stack_1:output:0;model/QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model/QF/ActivityRegularizer/strided_slice?
!model/QF/ActivityRegularizer/CastCast3model/QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!model/QF/ActivityRegularizer/Cast?
$model/QF/ActivityRegularizer/truedivRealDiv$model/QF/ActivityRegularizer/mul:z:0%model/QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$model/QF/ActivityRegularizer/truediv?
model/QB/MatMul/ReadVariableOpReadVariableOp'model_qb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/QB/MatMul/ReadVariableOp?
model/QB/MatMulMatMul!model/concatenate/concat:output:0&model/QB/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/QB/MatMul?
model/QB/BiasAdd/ReadVariableOpReadVariableOp(model_qb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
model/QB/BiasAdd/ReadVariableOp?
model/QB/BiasAddBiasAddmodel/QB/MatMul:product:0'model/QB/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/QB/BiasAdd|
model/QB/SigmoidSigmoidmodel/QB/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/QB/Sigmoid?
#model/QB/ActivityRegularizer/SquareSquaremodel/QB/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#model/QB/ActivityRegularizer/Square?
"model/QB/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model/QB/ActivityRegularizer/Const?
 model/QB/ActivityRegularizer/SumSum'model/QB/ActivityRegularizer/Square:y:0+model/QB/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 model/QB/ActivityRegularizer/Sum?
"model/QB/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"model/QB/ActivityRegularizer/mul/x?
 model/QB/ActivityRegularizer/mulMul+model/QB/ActivityRegularizer/mul/x:output:0)model/QB/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 model/QB/ActivityRegularizer/mul?
"model/QB/ActivityRegularizer/ShapeShapemodel/QB/Sigmoid:y:0*
T0*
_output_shapes
:2$
"model/QB/ActivityRegularizer/Shape?
0model/QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model/QB/ActivityRegularizer/strided_slice/stack?
2model/QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model/QB/ActivityRegularizer/strided_slice/stack_1?
2model/QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model/QB/ActivityRegularizer/strided_slice/stack_2?
*model/QB/ActivityRegularizer/strided_sliceStridedSlice+model/QB/ActivityRegularizer/Shape:output:09model/QB/ActivityRegularizer/strided_slice/stack:output:0;model/QB/ActivityRegularizer/strided_slice/stack_1:output:0;model/QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model/QB/ActivityRegularizer/strided_slice?
!model/QB/ActivityRegularizer/CastCast3model/QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!model/QB/ActivityRegularizer/Cast?
$model/QB/ActivityRegularizer/truedivRealDiv$model/QB/ActivityRegularizer/mul:z:0%model/QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$model/QB/ActivityRegularizer/truediv?
 model/STAT/MatMul/ReadVariableOpReadVariableOp)model_stat_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/STAT/MatMul/ReadVariableOp?
model/STAT/MatMulMatMul!model/concatenate/concat:output:0(model/STAT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/STAT/MatMul?
!model/STAT/BiasAdd/ReadVariableOpReadVariableOp*model_stat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/STAT/BiasAdd/ReadVariableOp?
model/STAT/BiasAddBiasAddmodel/STAT/MatMul:product:0)model/STAT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/STAT/BiasAdd?
model/STAT/SigmoidSigmoidmodel/STAT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/STAT/Sigmoid?
%model/STAT/ActivityRegularizer/SquareSquaremodel/STAT/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2'
%model/STAT/ActivityRegularizer/Square?
$model/STAT/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$model/STAT/ActivityRegularizer/Const?
"model/STAT/ActivityRegularizer/SumSum)model/STAT/ActivityRegularizer/Square:y:0-model/STAT/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2$
"model/STAT/ActivityRegularizer/Sum?
$model/STAT/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2&
$model/STAT/ActivityRegularizer/mul/x?
"model/STAT/ActivityRegularizer/mulMul-model/STAT/ActivityRegularizer/mul/x:output:0+model/STAT/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"model/STAT/ActivityRegularizer/mul?
$model/STAT/ActivityRegularizer/ShapeShapemodel/STAT/Sigmoid:y:0*
T0*
_output_shapes
:2&
$model/STAT/ActivityRegularizer/Shape?
2model/STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/STAT/ActivityRegularizer/strided_slice/stack?
4model/STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/STAT/ActivityRegularizer/strided_slice/stack_1?
4model/STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/STAT/ActivityRegularizer/strided_slice/stack_2?
,model/STAT/ActivityRegularizer/strided_sliceStridedSlice-model/STAT/ActivityRegularizer/Shape:output:0;model/STAT/ActivityRegularizer/strided_slice/stack:output:0=model/STAT/ActivityRegularizer/strided_slice/stack_1:output:0=model/STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model/STAT/ActivityRegularizer/strided_slice?
#model/STAT/ActivityRegularizer/CastCast5model/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#model/STAT/ActivityRegularizer/Cast?
&model/STAT/ActivityRegularizer/truedivRealDiv&model/STAT/ActivityRegularizer/mul:z:0'model/STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&model/STAT/ActivityRegularizer/truediv?
 model/MATH/MatMul/ReadVariableOpReadVariableOp)model_math_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 model/MATH/MatMul/ReadVariableOp?
model/MATH/MatMulMatMul!model/concatenate/concat:output:0(model/MATH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/MATH/MatMul?
!model/MATH/BiasAdd/ReadVariableOpReadVariableOp*model_math_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/MATH/BiasAdd/ReadVariableOp?
model/MATH/BiasAddBiasAddmodel/MATH/MatMul:product:0)model/MATH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/MATH/BiasAdd?
model/MATH/SigmoidSigmoidmodel/MATH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/MATH/Sigmoid?
%model/MATH/ActivityRegularizer/SquareSquaremodel/MATH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2'
%model/MATH/ActivityRegularizer/Square?
$model/MATH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$model/MATH/ActivityRegularizer/Const?
"model/MATH/ActivityRegularizer/SumSum)model/MATH/ActivityRegularizer/Square:y:0-model/MATH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2$
"model/MATH/ActivityRegularizer/Sum?
$model/MATH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2&
$model/MATH/ActivityRegularizer/mul/x?
"model/MATH/ActivityRegularizer/mulMul-model/MATH/ActivityRegularizer/mul/x:output:0+model/MATH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"model/MATH/ActivityRegularizer/mul?
$model/MATH/ActivityRegularizer/ShapeShapemodel/MATH/Sigmoid:y:0*
T0*
_output_shapes
:2&
$model/MATH/ActivityRegularizer/Shape?
2model/MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model/MATH/ActivityRegularizer/strided_slice/stack?
4model/MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model/MATH/ActivityRegularizer/strided_slice/stack_1?
4model/MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model/MATH/ActivityRegularizer/strided_slice/stack_2?
,model/MATH/ActivityRegularizer/strided_sliceStridedSlice-model/MATH/ActivityRegularizer/Shape:output:0;model/MATH/ActivityRegularizer/strided_slice/stack:output:0=model/MATH/ActivityRegularizer/strided_slice/stack_1:output:0=model/MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model/MATH/ActivityRegularizer/strided_slice?
#model/MATH/ActivityRegularizer/CastCast5model/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#model/MATH/ActivityRegularizer/Cast?
&model/MATH/ActivityRegularizer/truedivRealDiv&model/MATH/ActivityRegularizer/mul:z:0'model/MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&model/MATH/ActivityRegularizer/truediv?
model/PH/MatMul/ReadVariableOpReadVariableOp'model_ph_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/PH/MatMul/ReadVariableOp?
model/PH/MatMulMatMul!model/concatenate/concat:output:0&model/PH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/PH/MatMul?
model/PH/BiasAdd/ReadVariableOpReadVariableOp(model_ph_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
model/PH/BiasAdd/ReadVariableOp?
model/PH/BiasAddBiasAddmodel/PH/MatMul:product:0'model/PH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/PH/BiasAdd|
model/PH/SigmoidSigmoidmodel/PH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/PH/Sigmoid?
#model/PH/ActivityRegularizer/SquareSquaremodel/PH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#model/PH/ActivityRegularizer/Square?
"model/PH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model/PH/ActivityRegularizer/Const?
 model/PH/ActivityRegularizer/SumSum'model/PH/ActivityRegularizer/Square:y:0+model/PH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 model/PH/ActivityRegularizer/Sum?
"model/PH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"model/PH/ActivityRegularizer/mul/x?
 model/PH/ActivityRegularizer/mulMul+model/PH/ActivityRegularizer/mul/x:output:0)model/PH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 model/PH/ActivityRegularizer/mul?
"model/PH/ActivityRegularizer/ShapeShapemodel/PH/Sigmoid:y:0*
T0*
_output_shapes
:2$
"model/PH/ActivityRegularizer/Shape?
0model/PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model/PH/ActivityRegularizer/strided_slice/stack?
2model/PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model/PH/ActivityRegularizer/strided_slice/stack_1?
2model/PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model/PH/ActivityRegularizer/strided_slice/stack_2?
*model/PH/ActivityRegularizer/strided_sliceStridedSlice+model/PH/ActivityRegularizer/Shape:output:09model/PH/ActivityRegularizer/strided_slice/stack:output:0;model/PH/ActivityRegularizer/strided_slice/stack_1:output:0;model/PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model/PH/ActivityRegularizer/strided_slice?
!model/PH/ActivityRegularizer/CastCast3model/PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!model/PH/ActivityRegularizer/Cast?
$model/PH/ActivityRegularizer/truedivRealDiv$model/PH/ActivityRegularizer/mul:z:0%model/PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$model/PH/ActivityRegularizer/truediv?
model/CS/MatMul/ReadVariableOpReadVariableOp'model_cs_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/CS/MatMul/ReadVariableOp?
model/CS/MatMulMatMul!model/concatenate/concat:output:0&model/CS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/CS/MatMul?
model/CS/BiasAdd/ReadVariableOpReadVariableOp(model_cs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
model/CS/BiasAdd/ReadVariableOp?
model/CS/BiasAddBiasAddmodel/CS/MatMul:product:0'model/CS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/CS/BiasAdd|
model/CS/SigmoidSigmoidmodel/CS/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/CS/Sigmoid?
#model/CS/ActivityRegularizer/SquareSquaremodel/CS/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#model/CS/ActivityRegularizer/Square?
"model/CS/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model/CS/ActivityRegularizer/Const?
 model/CS/ActivityRegularizer/SumSum'model/CS/ActivityRegularizer/Square:y:0+model/CS/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 model/CS/ActivityRegularizer/Sum?
"model/CS/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"model/CS/ActivityRegularizer/mul/x?
 model/CS/ActivityRegularizer/mulMul+model/CS/ActivityRegularizer/mul/x:output:0)model/CS/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 model/CS/ActivityRegularizer/mul?
"model/CS/ActivityRegularizer/ShapeShapemodel/CS/Sigmoid:y:0*
T0*
_output_shapes
:2$
"model/CS/ActivityRegularizer/Shape?
0model/CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model/CS/ActivityRegularizer/strided_slice/stack?
2model/CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model/CS/ActivityRegularizer/strided_slice/stack_1?
2model/CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model/CS/ActivityRegularizer/strided_slice/stack_2?
*model/CS/ActivityRegularizer/strided_sliceStridedSlice+model/CS/ActivityRegularizer/Shape:output:09model/CS/ActivityRegularizer/strided_slice/stack:output:0;model/CS/ActivityRegularizer/strided_slice/stack_1:output:0;model/CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model/CS/ActivityRegularizer/strided_slice?
!model/CS/ActivityRegularizer/CastCast3model/CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!model/CS/ActivityRegularizer/Cast?
$model/CS/ActivityRegularizer/truedivRealDiv$model/CS/ActivityRegularizer/mul:z:0%model/CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$model/CS/ActivityRegularizer/truediv?
IdentityIdentitymodel/CS/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitymodel/MATH/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitymodel/PH/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitymodel/QB/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitymodel/QF/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identitymodel/STAT/Sigmoid:y:0 ^model/CS/BiasAdd/ReadVariableOp^model/CS/MatMul/ReadVariableOp"^model/MATH/BiasAdd/ReadVariableOp!^model/MATH/MatMul/ReadVariableOp ^model/PH/BiasAdd/ReadVariableOp^model/PH/MatMul/ReadVariableOp ^model/QB/BiasAdd/ReadVariableOp^model/QB/MatMul/ReadVariableOp ^model/QF/BiasAdd/ReadVariableOp^model/QF/MatMul/ReadVariableOp"^model/STAT/BiasAdd/ReadVariableOp!^model/STAT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::2B
model/CS/BiasAdd/ReadVariableOpmodel/CS/BiasAdd/ReadVariableOp2@
model/CS/MatMul/ReadVariableOpmodel/CS/MatMul/ReadVariableOp2F
!model/MATH/BiasAdd/ReadVariableOp!model/MATH/BiasAdd/ReadVariableOp2D
 model/MATH/MatMul/ReadVariableOp model/MATH/MatMul/ReadVariableOp2B
model/PH/BiasAdd/ReadVariableOpmodel/PH/BiasAdd/ReadVariableOp2@
model/PH/MatMul/ReadVariableOpmodel/PH/MatMul/ReadVariableOp2B
model/QB/BiasAdd/ReadVariableOpmodel/QB/BiasAdd/ReadVariableOp2@
model/QB/MatMul/ReadVariableOpmodel/QB/MatMul/ReadVariableOp2B
model/QF/BiasAdd/ReadVariableOpmodel/QF/BiasAdd/ReadVariableOp2@
model/QF/MatMul/ReadVariableOpmodel/QF/MatMul/ReadVariableOp2F
!model/STAT/BiasAdd/ReadVariableOp!model/STAT/BiasAdd/ReadVariableOp2D
 model/STAT/MatMul/ReadVariableOp model/STAT/MatMul/ReadVariableOp:P L
)
_output_shapes
:???????????

_user_specified_nametitle:SO
)
_output_shapes
:???????????
"
_user_specified_name
abstract
?

?
B__inference_CS_layer_call_and_return_all_conditional_losses_109990

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_CS_activity_regularizer_1084822
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>__inference_QF_layer_call_and_return_conditional_losses_110185

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+QF/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
,__inference_MATH_activity_regularizer_108508
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
?
&__inference_model_layer_call_fn_109888
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1092002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
&__inference_model_layer_call_fn_109934
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1093772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
@__inference_STAT_layer_call_and_return_conditional_losses_110099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-STAT/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_1102278
4ph_kernel_regularizer_square_readvariableop_resource
identity??+PH/kernel/Regularizer/Square/ReadVariableOp?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4ph_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
IdentityIdentityPH/kernel/Regularizer/mul:z:0,^PH/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_4_1102608
4qb_kernel_regularizer_square_readvariableop_resource
identity??+QB/kernel/Regularizer/Square/ReadVariableOp?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4qb_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
IdentityIdentityQB/kernel/Regularizer/mul:z:0,^QB/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp
?
?
>__inference_PH_layer_call_and_return_conditional_losses_108797

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+PH/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
ғ
?

A__inference_model_layer_call_and_return_conditional_losses_109674
inputs_0
inputs_1%
!qf_matmul_readvariableop_resource&
"qf_biasadd_readvariableop_resource%
!qb_matmul_readvariableop_resource&
"qb_biasadd_readvariableop_resource'
#stat_matmul_readvariableop_resource(
$stat_biasadd_readvariableop_resource'
#math_matmul_readvariableop_resource(
$math_biasadd_readvariableop_resource%
!ph_matmul_readvariableop_resource&
"ph_biasadd_readvariableop_resource%
!cs_matmul_readvariableop_resource&
"cs_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/BiasAdd/ReadVariableOp?CS/MatMul/ReadVariableOp?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/BiasAdd/ReadVariableOp?MATH/MatMul/ReadVariableOp?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/BiasAdd/ReadVariableOp?PH/MatMul/ReadVariableOp?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/BiasAdd/ReadVariableOp?QB/MatMul/ReadVariableOp?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/BiasAdd/ReadVariableOp?QF/MatMul/ReadVariableOp?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/BiasAdd/ReadVariableOp?STAT/MatMul/ReadVariableOp?-STAT/kernel/Regularizer/Square/ReadVariableOpt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*)
_output_shapes
:???????????2
concatenate/concat?
QF/MatMul/ReadVariableOpReadVariableOp!qf_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
QF/MatMul/ReadVariableOp?
	QF/MatMulMatMulconcatenate/concat:output:0 QF/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	QF/MatMul?
QF/BiasAdd/ReadVariableOpReadVariableOp"qf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
QF/BiasAdd/ReadVariableOp?

QF/BiasAddBiasAddQF/MatMul:product:0!QF/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

QF/BiasAddj

QF/SigmoidSigmoidQF/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

QF/Sigmoid?
QF/ActivityRegularizer/SquareSquareQF/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
QF/ActivityRegularizer/Square?
QF/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/ActivityRegularizer/Const?
QF/ActivityRegularizer/SumSum!QF/ActivityRegularizer/Square:y:0%QF/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/ActivityRegularizer/Sum?
QF/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/ActivityRegularizer/mul/x?
QF/ActivityRegularizer/mulMul%QF/ActivityRegularizer/mul/x:output:0#QF/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/ActivityRegularizer/mulz
QF/ActivityRegularizer/ShapeShapeQF/Sigmoid:y:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDivQF/ActivityRegularizer/mul:z:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/MatMul/ReadVariableOpReadVariableOp!qb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
QB/MatMul/ReadVariableOp?
	QB/MatMulMatMulconcatenate/concat:output:0 QB/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	QB/MatMul?
QB/BiasAdd/ReadVariableOpReadVariableOp"qb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
QB/BiasAdd/ReadVariableOp?

QB/BiasAddBiasAddQB/MatMul:product:0!QB/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

QB/BiasAddj

QB/SigmoidSigmoidQB/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

QB/Sigmoid?
QB/ActivityRegularizer/SquareSquareQB/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
QB/ActivityRegularizer/Square?
QB/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/ActivityRegularizer/Const?
QB/ActivityRegularizer/SumSum!QB/ActivityRegularizer/Square:y:0%QB/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/ActivityRegularizer/Sum?
QB/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/ActivityRegularizer/mul/x?
QB/ActivityRegularizer/mulMul%QB/ActivityRegularizer/mul/x:output:0#QB/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/ActivityRegularizer/mulz
QB/ActivityRegularizer/ShapeShapeQB/Sigmoid:y:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDivQB/ActivityRegularizer/mul:z:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/MatMul/ReadVariableOpReadVariableOp#stat_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
STAT/MatMul/ReadVariableOp?
STAT/MatMulMatMulconcatenate/concat:output:0"STAT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
STAT/MatMul?
STAT/BiasAdd/ReadVariableOpReadVariableOp$stat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
STAT/BiasAdd/ReadVariableOp?
STAT/BiasAddBiasAddSTAT/MatMul:product:0#STAT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
STAT/BiasAddp
STAT/SigmoidSigmoidSTAT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
STAT/Sigmoid?
STAT/ActivityRegularizer/SquareSquareSTAT/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2!
STAT/ActivityRegularizer/Square?
STAT/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
STAT/ActivityRegularizer/Const?
STAT/ActivityRegularizer/SumSum#STAT/ActivityRegularizer/Square:y:0'STAT/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/ActivityRegularizer/Sum?
STAT/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2 
STAT/ActivityRegularizer/mul/x?
STAT/ActivityRegularizer/mulMul'STAT/ActivityRegularizer/mul/x:output:0%STAT/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/ActivityRegularizer/mul?
STAT/ActivityRegularizer/ShapeShapeSTAT/Sigmoid:y:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv STAT/ActivityRegularizer/mul:z:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/MatMul/ReadVariableOpReadVariableOp#math_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MATH/MatMul/ReadVariableOp?
MATH/MatMulMatMulconcatenate/concat:output:0"MATH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MATH/MatMul?
MATH/BiasAdd/ReadVariableOpReadVariableOp$math_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
MATH/BiasAdd/ReadVariableOp?
MATH/BiasAddBiasAddMATH/MatMul:product:0#MATH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MATH/BiasAddp
MATH/SigmoidSigmoidMATH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
MATH/Sigmoid?
MATH/ActivityRegularizer/SquareSquareMATH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2!
MATH/ActivityRegularizer/Square?
MATH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
MATH/ActivityRegularizer/Const?
MATH/ActivityRegularizer/SumSum#MATH/ActivityRegularizer/Square:y:0'MATH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/ActivityRegularizer/Sum?
MATH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2 
MATH/ActivityRegularizer/mul/x?
MATH/ActivityRegularizer/mulMul'MATH/ActivityRegularizer/mul/x:output:0%MATH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/ActivityRegularizer/mul?
MATH/ActivityRegularizer/ShapeShapeMATH/Sigmoid:y:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv MATH/ActivityRegularizer/mul:z:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/MatMul/ReadVariableOpReadVariableOp!ph_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
PH/MatMul/ReadVariableOp?
	PH/MatMulMatMulconcatenate/concat:output:0 PH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	PH/MatMul?
PH/BiasAdd/ReadVariableOpReadVariableOp"ph_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
PH/BiasAdd/ReadVariableOp?

PH/BiasAddBiasAddPH/MatMul:product:0!PH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

PH/BiasAddj

PH/SigmoidSigmoidPH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

PH/Sigmoid?
PH/ActivityRegularizer/SquareSquarePH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
PH/ActivityRegularizer/Square?
PH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/ActivityRegularizer/Const?
PH/ActivityRegularizer/SumSum!PH/ActivityRegularizer/Square:y:0%PH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/ActivityRegularizer/Sum?
PH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/ActivityRegularizer/mul/x?
PH/ActivityRegularizer/mulMul%PH/ActivityRegularizer/mul/x:output:0#PH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/ActivityRegularizer/mulz
PH/ActivityRegularizer/ShapeShapePH/Sigmoid:y:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDivPH/ActivityRegularizer/mul:z:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/MatMul/ReadVariableOpReadVariableOp!cs_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
CS/MatMul/ReadVariableOp?
	CS/MatMulMatMulconcatenate/concat:output:0 CS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	CS/MatMul?
CS/BiasAdd/ReadVariableOpReadVariableOp"cs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
CS/BiasAdd/ReadVariableOp?

CS/BiasAddBiasAddCS/MatMul:product:0!CS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

CS/BiasAddj

CS/SigmoidSigmoidCS/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

CS/Sigmoid?
CS/ActivityRegularizer/SquareSquareCS/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
CS/ActivityRegularizer/Square?
CS/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/ActivityRegularizer/Const?
CS/ActivityRegularizer/SumSum!CS/ActivityRegularizer/Square:y:0%CS/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/ActivityRegularizer/Sum?
CS/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/ActivityRegularizer/mul/x?
CS/ActivityRegularizer/mulMul%CS/ActivityRegularizer/mul/x:output:0#CS/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/ActivityRegularizer/mulz
CS/ActivityRegularizer/ShapeShapeCS/Sigmoid:y:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDivCS/ActivityRegularizer/mul:z:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!cs_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!ph_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#math_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#stat_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!qb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!qf_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentityCS/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1IdentityPH/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2IdentityMATH/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3IdentitySTAT/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4IdentityQB/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5IdentityQF/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::26
CS/BiasAdd/ReadVariableOpCS/BiasAdd/ReadVariableOp24
CS/MatMul/ReadVariableOpCS/MatMul/ReadVariableOp2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2:
MATH/BiasAdd/ReadVariableOpMATH/BiasAdd/ReadVariableOp28
MATH/MatMul/ReadVariableOpMATH/MatMul/ReadVariableOp2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp26
PH/BiasAdd/ReadVariableOpPH/BiasAdd/ReadVariableOp24
PH/MatMul/ReadVariableOpPH/MatMul/ReadVariableOp2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp26
QB/BiasAdd/ReadVariableOpQB/BiasAdd/ReadVariableOp24
QB/MatMul/ReadVariableOpQB/MatMul/ReadVariableOp2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp26
QF/BiasAdd/ReadVariableOpQF/BiasAdd/ReadVariableOp24
QF/MatMul/ReadVariableOpQF/MatMul/ReadVariableOp2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2:
STAT/BiasAdd/ReadVariableOpSTAT/BiasAdd/ReadVariableOp28
STAT/MatMul/ReadVariableOpSTAT/MatMul/ReadVariableOp2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

?
D__inference_MATH_layer_call_and_return_all_conditional_losses_110076

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_MATH_activity_regularizer_1085082
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_MATH_layer_call_and_return_conditional_losses_108744

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?-MATH/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
x
#__inference_CS_layer_call_fn_109979

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>__inference_CS_layer_call_and_return_conditional_losses_108850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?+CS/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
X
,__inference_concatenate_layer_call_fn_109947
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1085592
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:???????????:???????????:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
$__inference_signature_wrapper_109506
abstract	
title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltitleabstractunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1084692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:???????????
"
_user_specified_name
abstract:PL
)
_output_shapes
:???????????

_user_specified_nametitle
?
z
%__inference_MATH_layer_call_fn_110065

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_QF_layer_call_and_return_all_conditional_losses_110205

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QF_activity_regularizer_1085472
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_109420	
title
abstract
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltitleabstractunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1093772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
)
_output_shapes
:???????????

_user_specified_nametitle:SO
)
_output_shapes
:???????????
"
_user_specified_name
abstract
?
?
__inference_loss_fn_0_1102168
4cs_kernel_regularizer_square_readvariableop_resource
identity??+CS/kernel/Regularizer/Square/ReadVariableOp?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4cs_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
IdentityIdentityCS/kernel/Regularizer/mul:z:0,^CS/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp
?
F
,__inference_STAT_activity_regularizer_108521
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
x
#__inference_PH_layer_call_fn_110022

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_STAT_layer_call_and_return_conditional_losses_108691

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-STAT/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_108934	
title
abstract
	qf_108608
	qf_108610
	qb_108661
	qb_108663
stat_108714
stat_108716
math_108767
math_108769
	ph_108820
	ph_108822
	cs_108873
	cs_108875
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/StatefulPartitionedCall?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/StatefulPartitionedCall?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/StatefulPartitionedCall?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/StatefulPartitionedCall?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/StatefulPartitionedCall?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/StatefulPartitionedCall?-STAT/kernel/Regularizer/Square/ReadVariableOp?
concatenate/PartitionedCallPartitionedCalltitleabstract*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1085592
concatenate/PartitionedCall?
QF/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qf_108608	qf_108610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
QF/StatefulPartitionedCall?
&QF/ActivityRegularizer/PartitionedCallPartitionedCall#QF/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QF_activity_regularizer_1085472(
&QF/ActivityRegularizer/PartitionedCall?
QF/ActivityRegularizer/ShapeShape#QF/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDiv/QF/ActivityRegularizer/PartitionedCall:output:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qb_108661	qb_108663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
QB/StatefulPartitionedCall?
&QB/ActivityRegularizer/PartitionedCallPartitionedCall#QB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QB_activity_regularizer_1085342(
&QB/ActivityRegularizer/PartitionedCall?
QB/ActivityRegularizer/ShapeShape#QB/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDiv/QB/ActivityRegularizer/PartitionedCall:output:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0stat_108714stat_108716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
STAT/StatefulPartitionedCall?
(STAT/ActivityRegularizer/PartitionedCallPartitionedCall%STAT/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_STAT_activity_regularizer_1085212*
(STAT/ActivityRegularizer/PartitionedCall?
STAT/ActivityRegularizer/ShapeShape%STAT/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv1STAT/ActivityRegularizer/PartitionedCall:output:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0math_108767math_108769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
MATH/StatefulPartitionedCall?
(MATH/ActivityRegularizer/PartitionedCallPartitionedCall%MATH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_MATH_activity_regularizer_1085082*
(MATH/ActivityRegularizer/PartitionedCall?
MATH/ActivityRegularizer/ShapeShape%MATH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv1MATH/ActivityRegularizer/PartitionedCall:output:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	ph_108820	ph_108822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
PH/StatefulPartitionedCall?
&PH/ActivityRegularizer/PartitionedCallPartitionedCall#PH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_PH_activity_regularizer_1084952(
&PH/ActivityRegularizer/PartitionedCall?
PH/ActivityRegularizer/ShapeShape#PH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDiv/PH/ActivityRegularizer/PartitionedCall:output:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	cs_108873	cs_108875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
CS/StatefulPartitionedCall?
&CS/ActivityRegularizer/PartitionedCallPartitionedCall#CS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_CS_activity_regularizer_1084822(
&CS/ActivityRegularizer/PartitionedCall?
CS/ActivityRegularizer/ShapeShape#CS/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDiv/CS/ActivityRegularizer/PartitionedCall:output:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	cs_108873* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	ph_108820* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmath_108767* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstat_108714* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qb_108661* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qf_108608* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentity#CS/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#PH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%MATH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity%STAT/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#QB/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#QF/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::28
CS/StatefulPartitionedCallCS/StatefulPartitionedCall2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2<
MATH/StatefulPartitionedCallMATH/StatefulPartitionedCall2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp28
PH/StatefulPartitionedCallPH/StatefulPartitionedCall2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp28
QB/StatefulPartitionedCallQB/StatefulPartitionedCall2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp28
QF/StatefulPartitionedCallQF/StatefulPartitionedCall2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2<
STAT/StatefulPartitionedCallSTAT/StatefulPartitionedCall2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:P L
)
_output_shapes
:???????????

_user_specified_nametitle:SO
)
_output_shapes
:???????????
"
_user_specified_name
abstract
?
?
>__inference_PH_layer_call_and_return_conditional_losses_110013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+PH/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
*__inference_CS_activity_regularizer_108482
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
??
?
A__inference_model_layer_call_and_return_conditional_losses_109200

inputs
inputs_1
	qf_109074
	qf_109076
	qb_109087
	qb_109089
stat_109100
stat_109102
math_109113
math_109115
	ph_109126
	ph_109128
	cs_109139
	cs_109141
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/StatefulPartitionedCall?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/StatefulPartitionedCall?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/StatefulPartitionedCall?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/StatefulPartitionedCall?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/StatefulPartitionedCall?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/StatefulPartitionedCall?-STAT/kernel/Regularizer/Square/ReadVariableOp?
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1085592
concatenate/PartitionedCall?
QF/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qf_109074	qf_109076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
QF/StatefulPartitionedCall?
&QF/ActivityRegularizer/PartitionedCallPartitionedCall#QF/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QF_activity_regularizer_1085472(
&QF/ActivityRegularizer/PartitionedCall?
QF/ActivityRegularizer/ShapeShape#QF/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDiv/QF/ActivityRegularizer/PartitionedCall:output:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qb_109087	qb_109089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
QB/StatefulPartitionedCall?
&QB/ActivityRegularizer/PartitionedCallPartitionedCall#QB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QB_activity_regularizer_1085342(
&QB/ActivityRegularizer/PartitionedCall?
QB/ActivityRegularizer/ShapeShape#QB/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDiv/QB/ActivityRegularizer/PartitionedCall:output:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0stat_109100stat_109102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
STAT/StatefulPartitionedCall?
(STAT/ActivityRegularizer/PartitionedCallPartitionedCall%STAT/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_STAT_activity_regularizer_1085212*
(STAT/ActivityRegularizer/PartitionedCall?
STAT/ActivityRegularizer/ShapeShape%STAT/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv1STAT/ActivityRegularizer/PartitionedCall:output:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0math_109113math_109115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
MATH/StatefulPartitionedCall?
(MATH/ActivityRegularizer/PartitionedCallPartitionedCall%MATH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_MATH_activity_regularizer_1085082*
(MATH/ActivityRegularizer/PartitionedCall?
MATH/ActivityRegularizer/ShapeShape%MATH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv1MATH/ActivityRegularizer/PartitionedCall:output:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	ph_109126	ph_109128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
PH/StatefulPartitionedCall?
&PH/ActivityRegularizer/PartitionedCallPartitionedCall#PH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_PH_activity_regularizer_1084952(
&PH/ActivityRegularizer/PartitionedCall?
PH/ActivityRegularizer/ShapeShape#PH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDiv/PH/ActivityRegularizer/PartitionedCall:output:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	cs_109139	cs_109141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
CS/StatefulPartitionedCall?
&CS/ActivityRegularizer/PartitionedCallPartitionedCall#CS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_CS_activity_regularizer_1084822(
&CS/ActivityRegularizer/PartitionedCall?
CS/ActivityRegularizer/ShapeShape#CS/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDiv/CS/ActivityRegularizer/PartitionedCall:output:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	cs_109139* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	ph_109126* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmath_109113* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstat_109100* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qb_109087* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qf_109074* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentity#CS/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#PH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%MATH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity%STAT/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#QB/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#QF/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::28
CS/StatefulPartitionedCallCS/StatefulPartitionedCall2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2<
MATH/StatefulPartitionedCallMATH/StatefulPartitionedCall2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp28
PH/StatefulPartitionedCallPH/StatefulPartitionedCall2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp28
QB/StatefulPartitionedCallQB/StatefulPartitionedCall2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp28
QF/StatefulPartitionedCallQF/StatefulPartitionedCall2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2<
STAT/StatefulPartitionedCallSTAT/StatefulPartitionedCall2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:QM
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_1102718
4qf_kernel_regularizer_square_readvariableop_resource
identity??+QF/kernel/Regularizer/Square/ReadVariableOp?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4qf_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentityQF/kernel/Regularizer/mul:z:0,^QF/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp
?
s
G__inference_concatenate_layer_call_and_return_conditional_losses_109941
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*)
_output_shapes
:???????????2
concate
IdentityIdentityconcat:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:???????????:???????????:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
D
*__inference_QF_activity_regularizer_108547
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
x
#__inference_QF_layer_call_fn_110194

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_109243	
title
abstract
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2

identity_3

identity_4

identity_5??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltitleabstractunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
~:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1092002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
)
_output_shapes
:???????????

_user_specified_nametitle:SO
)
_output_shapes
:???????????
"
_user_specified_name
abstract
?
?
@__inference_MATH_layer_call_and_return_conditional_losses_110056

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?-MATH/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_QB_layer_call_and_return_all_conditional_losses_110162

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QB_activity_regularizer_1085342
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_110249:
6stat_kernel_regularizer_square_readvariableop_resource
identity??-STAT/kernel/Regularizer/Square/ReadVariableOp?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6stat_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
IdentityIdentitySTAT/kernel/Regularizer/mul:z:0.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp
ғ
?

A__inference_model_layer_call_and_return_conditional_losses_109842
inputs_0
inputs_1%
!qf_matmul_readvariableop_resource&
"qf_biasadd_readvariableop_resource%
!qb_matmul_readvariableop_resource&
"qb_biasadd_readvariableop_resource'
#stat_matmul_readvariableop_resource(
$stat_biasadd_readvariableop_resource'
#math_matmul_readvariableop_resource(
$math_biasadd_readvariableop_resource%
!ph_matmul_readvariableop_resource&
"ph_biasadd_readvariableop_resource%
!cs_matmul_readvariableop_resource&
"cs_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/BiasAdd/ReadVariableOp?CS/MatMul/ReadVariableOp?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/BiasAdd/ReadVariableOp?MATH/MatMul/ReadVariableOp?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/BiasAdd/ReadVariableOp?PH/MatMul/ReadVariableOp?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/BiasAdd/ReadVariableOp?QB/MatMul/ReadVariableOp?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/BiasAdd/ReadVariableOp?QF/MatMul/ReadVariableOp?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/BiasAdd/ReadVariableOp?STAT/MatMul/ReadVariableOp?-STAT/kernel/Regularizer/Square/ReadVariableOpt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*)
_output_shapes
:???????????2
concatenate/concat?
QF/MatMul/ReadVariableOpReadVariableOp!qf_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
QF/MatMul/ReadVariableOp?
	QF/MatMulMatMulconcatenate/concat:output:0 QF/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	QF/MatMul?
QF/BiasAdd/ReadVariableOpReadVariableOp"qf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
QF/BiasAdd/ReadVariableOp?

QF/BiasAddBiasAddQF/MatMul:product:0!QF/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

QF/BiasAddj

QF/SigmoidSigmoidQF/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

QF/Sigmoid?
QF/ActivityRegularizer/SquareSquareQF/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
QF/ActivityRegularizer/Square?
QF/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/ActivityRegularizer/Const?
QF/ActivityRegularizer/SumSum!QF/ActivityRegularizer/Square:y:0%QF/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/ActivityRegularizer/Sum?
QF/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/ActivityRegularizer/mul/x?
QF/ActivityRegularizer/mulMul%QF/ActivityRegularizer/mul/x:output:0#QF/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/ActivityRegularizer/mulz
QF/ActivityRegularizer/ShapeShapeQF/Sigmoid:y:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDivQF/ActivityRegularizer/mul:z:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/MatMul/ReadVariableOpReadVariableOp!qb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
QB/MatMul/ReadVariableOp?
	QB/MatMulMatMulconcatenate/concat:output:0 QB/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	QB/MatMul?
QB/BiasAdd/ReadVariableOpReadVariableOp"qb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
QB/BiasAdd/ReadVariableOp?

QB/BiasAddBiasAddQB/MatMul:product:0!QB/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

QB/BiasAddj

QB/SigmoidSigmoidQB/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

QB/Sigmoid?
QB/ActivityRegularizer/SquareSquareQB/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
QB/ActivityRegularizer/Square?
QB/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/ActivityRegularizer/Const?
QB/ActivityRegularizer/SumSum!QB/ActivityRegularizer/Square:y:0%QB/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/ActivityRegularizer/Sum?
QB/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/ActivityRegularizer/mul/x?
QB/ActivityRegularizer/mulMul%QB/ActivityRegularizer/mul/x:output:0#QB/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/ActivityRegularizer/mulz
QB/ActivityRegularizer/ShapeShapeQB/Sigmoid:y:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDivQB/ActivityRegularizer/mul:z:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/MatMul/ReadVariableOpReadVariableOp#stat_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
STAT/MatMul/ReadVariableOp?
STAT/MatMulMatMulconcatenate/concat:output:0"STAT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
STAT/MatMul?
STAT/BiasAdd/ReadVariableOpReadVariableOp$stat_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
STAT/BiasAdd/ReadVariableOp?
STAT/BiasAddBiasAddSTAT/MatMul:product:0#STAT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
STAT/BiasAddp
STAT/SigmoidSigmoidSTAT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
STAT/Sigmoid?
STAT/ActivityRegularizer/SquareSquareSTAT/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2!
STAT/ActivityRegularizer/Square?
STAT/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
STAT/ActivityRegularizer/Const?
STAT/ActivityRegularizer/SumSum#STAT/ActivityRegularizer/Square:y:0'STAT/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/ActivityRegularizer/Sum?
STAT/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2 
STAT/ActivityRegularizer/mul/x?
STAT/ActivityRegularizer/mulMul'STAT/ActivityRegularizer/mul/x:output:0%STAT/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/ActivityRegularizer/mul?
STAT/ActivityRegularizer/ShapeShapeSTAT/Sigmoid:y:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv STAT/ActivityRegularizer/mul:z:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/MatMul/ReadVariableOpReadVariableOp#math_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MATH/MatMul/ReadVariableOp?
MATH/MatMulMatMulconcatenate/concat:output:0"MATH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MATH/MatMul?
MATH/BiasAdd/ReadVariableOpReadVariableOp$math_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
MATH/BiasAdd/ReadVariableOp?
MATH/BiasAddBiasAddMATH/MatMul:product:0#MATH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MATH/BiasAddp
MATH/SigmoidSigmoidMATH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
MATH/Sigmoid?
MATH/ActivityRegularizer/SquareSquareMATH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2!
MATH/ActivityRegularizer/Square?
MATH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
MATH/ActivityRegularizer/Const?
MATH/ActivityRegularizer/SumSum#MATH/ActivityRegularizer/Square:y:0'MATH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/ActivityRegularizer/Sum?
MATH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2 
MATH/ActivityRegularizer/mul/x?
MATH/ActivityRegularizer/mulMul'MATH/ActivityRegularizer/mul/x:output:0%MATH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/ActivityRegularizer/mul?
MATH/ActivityRegularizer/ShapeShapeMATH/Sigmoid:y:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv MATH/ActivityRegularizer/mul:z:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/MatMul/ReadVariableOpReadVariableOp!ph_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
PH/MatMul/ReadVariableOp?
	PH/MatMulMatMulconcatenate/concat:output:0 PH/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	PH/MatMul?
PH/BiasAdd/ReadVariableOpReadVariableOp"ph_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
PH/BiasAdd/ReadVariableOp?

PH/BiasAddBiasAddPH/MatMul:product:0!PH/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

PH/BiasAddj

PH/SigmoidSigmoidPH/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

PH/Sigmoid?
PH/ActivityRegularizer/SquareSquarePH/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
PH/ActivityRegularizer/Square?
PH/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/ActivityRegularizer/Const?
PH/ActivityRegularizer/SumSum!PH/ActivityRegularizer/Square:y:0%PH/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/ActivityRegularizer/Sum?
PH/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/ActivityRegularizer/mul/x?
PH/ActivityRegularizer/mulMul%PH/ActivityRegularizer/mul/x:output:0#PH/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/ActivityRegularizer/mulz
PH/ActivityRegularizer/ShapeShapePH/Sigmoid:y:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDivPH/ActivityRegularizer/mul:z:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/MatMul/ReadVariableOpReadVariableOp!cs_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
CS/MatMul/ReadVariableOp?
	CS/MatMulMatMulconcatenate/concat:output:0 CS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	CS/MatMul?
CS/BiasAdd/ReadVariableOpReadVariableOp"cs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
CS/BiasAdd/ReadVariableOp?

CS/BiasAddBiasAddCS/MatMul:product:0!CS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

CS/BiasAddj

CS/SigmoidSigmoidCS/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

CS/Sigmoid?
CS/ActivityRegularizer/SquareSquareCS/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
CS/ActivityRegularizer/Square?
CS/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/ActivityRegularizer/Const?
CS/ActivityRegularizer/SumSum!CS/ActivityRegularizer/Square:y:0%CS/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/ActivityRegularizer/Sum?
CS/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/ActivityRegularizer/mul/x?
CS/ActivityRegularizer/mulMul%CS/ActivityRegularizer/mul/x:output:0#CS/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/ActivityRegularizer/mulz
CS/ActivityRegularizer/ShapeShapeCS/Sigmoid:y:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDivCS/ActivityRegularizer/mul:z:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!cs_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!ph_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#math_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#stat_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!qb_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!qf_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentityCS/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1IdentityPH/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2IdentityMATH/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3IdentitySTAT/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4IdentityQB/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5IdentityQF/Sigmoid:y:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/BiasAdd/ReadVariableOp^CS/MatMul/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/BiasAdd/ReadVariableOp^MATH/MatMul/ReadVariableOp.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/BiasAdd/ReadVariableOp^PH/MatMul/ReadVariableOp,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/BiasAdd/ReadVariableOp^QB/MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/BiasAdd/ReadVariableOp^QF/MatMul/ReadVariableOp,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/BiasAdd/ReadVariableOp^STAT/MatMul/ReadVariableOp.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::26
CS/BiasAdd/ReadVariableOpCS/BiasAdd/ReadVariableOp24
CS/MatMul/ReadVariableOpCS/MatMul/ReadVariableOp2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2:
MATH/BiasAdd/ReadVariableOpMATH/BiasAdd/ReadVariableOp28
MATH/MatMul/ReadVariableOpMATH/MatMul/ReadVariableOp2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp26
PH/BiasAdd/ReadVariableOpPH/BiasAdd/ReadVariableOp24
PH/MatMul/ReadVariableOpPH/MatMul/ReadVariableOp2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp26
QB/BiasAdd/ReadVariableOpQB/BiasAdd/ReadVariableOp24
QB/MatMul/ReadVariableOpQB/MatMul/ReadVariableOp2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp26
QF/BiasAdd/ReadVariableOpQF/BiasAdd/ReadVariableOp24
QF/MatMul/ReadVariableOpQF/MatMul/ReadVariableOp2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2:
STAT/BiasAdd/ReadVariableOpSTAT/BiasAdd/ReadVariableOp28
STAT/MatMul/ReadVariableOpSTAT/MatMul/ReadVariableOp2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:S O
)
_output_shapes
:???????????
"
_user_specified_name
inputs/0:SO
)
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
z
%__inference_STAT_layer_call_fn_110108

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_STAT_layer_call_and_return_all_conditional_losses_110119

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_STAT_activity_regularizer_1085212
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
*__inference_QB_activity_regularizer_108534
self
identityC
SquareSquareself*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
?
q
G__inference_concatenate_layer_call_and_return_conditional_losses_108559

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*)
_output_shapes
:???????????2
concate
IdentityIdentityconcat:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:???????????:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:QM
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_PH_layer_call_and_return_all_conditional_losses_110033

inputs
unknown
	unknown_0
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_PH_activity_regularizer_1084952
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
x
#__inference_QB_layer_call_fn_110151

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
>__inference_CS_layer_call_and_return_conditional_losses_109970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?+CS/kernel/Regularizer/Square/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp,^CS/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_109065	
title
abstract
	qf_108939
	qf_108941
	qb_108952
	qb_108954
stat_108965
stat_108967
math_108978
math_108980
	ph_108991
	ph_108993
	cs_109004
	cs_109006
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??CS/StatefulPartitionedCall?+CS/kernel/Regularizer/Square/ReadVariableOp?MATH/StatefulPartitionedCall?-MATH/kernel/Regularizer/Square/ReadVariableOp?PH/StatefulPartitionedCall?+PH/kernel/Regularizer/Square/ReadVariableOp?QB/StatefulPartitionedCall?+QB/kernel/Regularizer/Square/ReadVariableOp?QF/StatefulPartitionedCall?+QF/kernel/Regularizer/Square/ReadVariableOp?STAT/StatefulPartitionedCall?-STAT/kernel/Regularizer/Square/ReadVariableOp?
concatenate/PartitionedCallPartitionedCalltitleabstract*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1085592
concatenate/PartitionedCall?
QF/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qf_108939	qf_108941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QF_layer_call_and_return_conditional_losses_1085852
QF/StatefulPartitionedCall?
&QF/ActivityRegularizer/PartitionedCallPartitionedCall#QF/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QF_activity_regularizer_1085472(
&QF/ActivityRegularizer/PartitionedCall?
QF/ActivityRegularizer/ShapeShape#QF/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QF/ActivityRegularizer/Shape?
*QF/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QF/ActivityRegularizer/strided_slice/stack?
,QF/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_1?
,QF/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QF/ActivityRegularizer/strided_slice/stack_2?
$QF/ActivityRegularizer/strided_sliceStridedSlice%QF/ActivityRegularizer/Shape:output:03QF/ActivityRegularizer/strided_slice/stack:output:05QF/ActivityRegularizer/strided_slice/stack_1:output:05QF/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QF/ActivityRegularizer/strided_slice?
QF/ActivityRegularizer/CastCast-QF/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QF/ActivityRegularizer/Cast?
QF/ActivityRegularizer/truedivRealDiv/QF/ActivityRegularizer/PartitionedCall:output:0QF/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QF/ActivityRegularizer/truediv?
QB/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	qb_108952	qb_108954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_QB_layer_call_and_return_conditional_losses_1086382
QB/StatefulPartitionedCall?
&QB/ActivityRegularizer/PartitionedCallPartitionedCall#QB/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_QB_activity_regularizer_1085342(
&QB/ActivityRegularizer/PartitionedCall?
QB/ActivityRegularizer/ShapeShape#QB/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
QB/ActivityRegularizer/Shape?
*QB/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*QB/ActivityRegularizer/strided_slice/stack?
,QB/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_1?
,QB/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,QB/ActivityRegularizer/strided_slice/stack_2?
$QB/ActivityRegularizer/strided_sliceStridedSlice%QB/ActivityRegularizer/Shape:output:03QB/ActivityRegularizer/strided_slice/stack:output:05QB/ActivityRegularizer/strided_slice/stack_1:output:05QB/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$QB/ActivityRegularizer/strided_slice?
QB/ActivityRegularizer/CastCast-QB/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
QB/ActivityRegularizer/Cast?
QB/ActivityRegularizer/truedivRealDiv/QB/ActivityRegularizer/PartitionedCall:output:0QB/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
QB/ActivityRegularizer/truediv?
STAT/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0stat_108965stat_108967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_STAT_layer_call_and_return_conditional_losses_1086912
STAT/StatefulPartitionedCall?
(STAT/ActivityRegularizer/PartitionedCallPartitionedCall%STAT/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_STAT_activity_regularizer_1085212*
(STAT/ActivityRegularizer/PartitionedCall?
STAT/ActivityRegularizer/ShapeShape%STAT/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
STAT/ActivityRegularizer/Shape?
,STAT/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,STAT/ActivityRegularizer/strided_slice/stack?
.STAT/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_1?
.STAT/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.STAT/ActivityRegularizer/strided_slice/stack_2?
&STAT/ActivityRegularizer/strided_sliceStridedSlice'STAT/ActivityRegularizer/Shape:output:05STAT/ActivityRegularizer/strided_slice/stack:output:07STAT/ActivityRegularizer/strided_slice/stack_1:output:07STAT/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&STAT/ActivityRegularizer/strided_slice?
STAT/ActivityRegularizer/CastCast/STAT/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
STAT/ActivityRegularizer/Cast?
 STAT/ActivityRegularizer/truedivRealDiv1STAT/ActivityRegularizer/PartitionedCall:output:0!STAT/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 STAT/ActivityRegularizer/truediv?
MATH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0math_108978math_108980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_MATH_layer_call_and_return_conditional_losses_1087442
MATH/StatefulPartitionedCall?
(MATH/ActivityRegularizer/PartitionedCallPartitionedCall%MATH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_MATH_activity_regularizer_1085082*
(MATH/ActivityRegularizer/PartitionedCall?
MATH/ActivityRegularizer/ShapeShape%MATH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2 
MATH/ActivityRegularizer/Shape?
,MATH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,MATH/ActivityRegularizer/strided_slice/stack?
.MATH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_1?
.MATH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.MATH/ActivityRegularizer/strided_slice/stack_2?
&MATH/ActivityRegularizer/strided_sliceStridedSlice'MATH/ActivityRegularizer/Shape:output:05MATH/ActivityRegularizer/strided_slice/stack:output:07MATH/ActivityRegularizer/strided_slice/stack_1:output:07MATH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&MATH/ActivityRegularizer/strided_slice?
MATH/ActivityRegularizer/CastCast/MATH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
MATH/ActivityRegularizer/Cast?
 MATH/ActivityRegularizer/truedivRealDiv1MATH/ActivityRegularizer/PartitionedCall:output:0!MATH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2"
 MATH/ActivityRegularizer/truediv?
PH/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	ph_108991	ph_108993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_PH_layer_call_and_return_conditional_losses_1087972
PH/StatefulPartitionedCall?
&PH/ActivityRegularizer/PartitionedCallPartitionedCall#PH/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_PH_activity_regularizer_1084952(
&PH/ActivityRegularizer/PartitionedCall?
PH/ActivityRegularizer/ShapeShape#PH/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
PH/ActivityRegularizer/Shape?
*PH/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*PH/ActivityRegularizer/strided_slice/stack?
,PH/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_1?
,PH/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,PH/ActivityRegularizer/strided_slice/stack_2?
$PH/ActivityRegularizer/strided_sliceStridedSlice%PH/ActivityRegularizer/Shape:output:03PH/ActivityRegularizer/strided_slice/stack:output:05PH/ActivityRegularizer/strided_slice/stack_1:output:05PH/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$PH/ActivityRegularizer/strided_slice?
PH/ActivityRegularizer/CastCast-PH/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
PH/ActivityRegularizer/Cast?
PH/ActivityRegularizer/truedivRealDiv/PH/ActivityRegularizer/PartitionedCall:output:0PH/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
PH/ActivityRegularizer/truediv?
CS/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0	cs_109004	cs_109006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CS_layer_call_and_return_conditional_losses_1088502
CS/StatefulPartitionedCall?
&CS/ActivityRegularizer/PartitionedCallPartitionedCall#CS/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_CS_activity_regularizer_1084822(
&CS/ActivityRegularizer/PartitionedCall?
CS/ActivityRegularizer/ShapeShape#CS/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
CS/ActivityRegularizer/Shape?
*CS/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*CS/ActivityRegularizer/strided_slice/stack?
,CS/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_1?
,CS/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,CS/ActivityRegularizer/strided_slice/stack_2?
$CS/ActivityRegularizer/strided_sliceStridedSlice%CS/ActivityRegularizer/Shape:output:03CS/ActivityRegularizer/strided_slice/stack:output:05CS/ActivityRegularizer/strided_slice/stack_1:output:05CS/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$CS/ActivityRegularizer/strided_slice?
CS/ActivityRegularizer/CastCast-CS/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
CS/ActivityRegularizer/Cast?
CS/ActivityRegularizer/truedivRealDiv/CS/ActivityRegularizer/PartitionedCall:output:0CS/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2 
CS/ActivityRegularizer/truediv?
+CS/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	cs_109004* 
_output_shapes
:
??*
dtype02-
+CS/kernel/Regularizer/Square/ReadVariableOp?
CS/kernel/Regularizer/SquareSquare3CS/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
CS/kernel/Regularizer/Square?
CS/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
CS/kernel/Regularizer/Const?
CS/kernel/Regularizer/SumSum CS/kernel/Regularizer/Square:y:0$CS/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/Sum
CS/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
CS/kernel/Regularizer/mul/x?
CS/kernel/Regularizer/mulMul$CS/kernel/Regularizer/mul/x:output:0"CS/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
CS/kernel/Regularizer/mul?
+PH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	ph_108991* 
_output_shapes
:
??*
dtype02-
+PH/kernel/Regularizer/Square/ReadVariableOp?
PH/kernel/Regularizer/SquareSquare3PH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
PH/kernel/Regularizer/Square?
PH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
PH/kernel/Regularizer/Const?
PH/kernel/Regularizer/SumSum PH/kernel/Regularizer/Square:y:0$PH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/Sum
PH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
PH/kernel/Regularizer/mul/x?
PH/kernel/Regularizer/mulMul$PH/kernel/Regularizer/mul/x:output:0"PH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
PH/kernel/Regularizer/mul?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmath_108978* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
-STAT/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstat_108965* 
_output_shapes
:
??*
dtype02/
-STAT/kernel/Regularizer/Square/ReadVariableOp?
STAT/kernel/Regularizer/SquareSquare5STAT/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
STAT/kernel/Regularizer/Square?
STAT/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
STAT/kernel/Regularizer/Const?
STAT/kernel/Regularizer/SumSum"STAT/kernel/Regularizer/Square:y:0&STAT/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/Sum?
STAT/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
STAT/kernel/Regularizer/mul/x?
STAT/kernel/Regularizer/mulMul&STAT/kernel/Regularizer/mul/x:output:0$STAT/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
STAT/kernel/Regularizer/mul?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qb_108952* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
+QF/kernel/Regularizer/Square/ReadVariableOpReadVariableOp	qf_108939* 
_output_shapes
:
??*
dtype02-
+QF/kernel/Regularizer/Square/ReadVariableOp?
QF/kernel/Regularizer/SquareSquare3QF/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QF/kernel/Regularizer/Square?
QF/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QF/kernel/Regularizer/Const?
QF/kernel/Regularizer/SumSum QF/kernel/Regularizer/Square:y:0$QF/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/Sum
QF/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QF/kernel/Regularizer/mul/x?
QF/kernel/Regularizer/mulMul$QF/kernel/Regularizer/mul/x:output:0"QF/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QF/kernel/Regularizer/mul?
IdentityIdentity#CS/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#PH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity%MATH/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity%STAT/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#QB/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#QF/StatefulPartitionedCall:output:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity"CS/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_6?

Identity_7Identity"PH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_7?

Identity_8Identity$MATH/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_8?

Identity_9Identity$STAT/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_9?
Identity_10Identity"QB/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_10?
Identity_11Identity"QF/ActivityRegularizer/truediv:z:0^CS/StatefulPartitionedCall,^CS/kernel/Regularizer/Square/ReadVariableOp^MATH/StatefulPartitionedCall.^MATH/kernel/Regularizer/Square/ReadVariableOp^PH/StatefulPartitionedCall,^PH/kernel/Regularizer/Square/ReadVariableOp^QB/StatefulPartitionedCall,^QB/kernel/Regularizer/Square/ReadVariableOp^QF/StatefulPartitionedCall,^QF/kernel/Regularizer/Square/ReadVariableOp^STAT/StatefulPartitionedCall.^STAT/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*m
_input_shapes\
Z:???????????:???????????::::::::::::28
CS/StatefulPartitionedCallCS/StatefulPartitionedCall2Z
+CS/kernel/Regularizer/Square/ReadVariableOp+CS/kernel/Regularizer/Square/ReadVariableOp2<
MATH/StatefulPartitionedCallMATH/StatefulPartitionedCall2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp28
PH/StatefulPartitionedCallPH/StatefulPartitionedCall2Z
+PH/kernel/Regularizer/Square/ReadVariableOp+PH/kernel/Regularizer/Square/ReadVariableOp28
QB/StatefulPartitionedCallQB/StatefulPartitionedCall2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp28
QF/StatefulPartitionedCallQF/StatefulPartitionedCall2Z
+QF/kernel/Regularizer/Square/ReadVariableOp+QF/kernel/Regularizer/Square/ReadVariableOp2<
STAT/StatefulPartitionedCallSTAT/StatefulPartitionedCall2^
-STAT/kernel/Regularizer/Square/ReadVariableOp-STAT/kernel/Regularizer/Square/ReadVariableOp:P L
)
_output_shapes
:???????????

_user_specified_nametitle:SO
)
_output_shapes
:???????????
"
_user_specified_name
abstract
??
?
"__inference__traced_restore_110712
file_prefix
assignvariableop_cs_kernel
assignvariableop_1_cs_bias 
assignvariableop_2_ph_kernel
assignvariableop_3_ph_bias"
assignvariableop_4_math_kernel 
assignvariableop_5_math_bias"
assignvariableop_6_stat_kernel 
assignvariableop_7_stat_bias 
assignvariableop_8_qb_kernel
assignvariableop_9_qb_bias!
assignvariableop_10_qf_kernel
assignvariableop_11_qf_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1
assignvariableop_21_total_2
assignvariableop_22_count_2
assignvariableop_23_total_3
assignvariableop_24_count_3
assignvariableop_25_total_4
assignvariableop_26_count_4
assignvariableop_27_total_5
assignvariableop_28_count_5
assignvariableop_29_total_6
assignvariableop_30_count_6
assignvariableop_31_total_7
assignvariableop_32_count_7
assignvariableop_33_total_8
assignvariableop_34_count_8
assignvariableop_35_total_9
assignvariableop_36_count_9 
assignvariableop_37_total_10 
assignvariableop_38_count_10 
assignvariableop_39_total_11 
assignvariableop_40_count_11 
assignvariableop_41_total_12 
assignvariableop_42_count_12(
$assignvariableop_43_adam_cs_kernel_m&
"assignvariableop_44_adam_cs_bias_m(
$assignvariableop_45_adam_ph_kernel_m&
"assignvariableop_46_adam_ph_bias_m*
&assignvariableop_47_adam_math_kernel_m(
$assignvariableop_48_adam_math_bias_m*
&assignvariableop_49_adam_stat_kernel_m(
$assignvariableop_50_adam_stat_bias_m(
$assignvariableop_51_adam_qb_kernel_m&
"assignvariableop_52_adam_qb_bias_m(
$assignvariableop_53_adam_qf_kernel_m&
"assignvariableop_54_adam_qf_bias_m(
$assignvariableop_55_adam_cs_kernel_v&
"assignvariableop_56_adam_cs_bias_v(
$assignvariableop_57_adam_ph_kernel_v&
"assignvariableop_58_adam_ph_bias_v*
&assignvariableop_59_adam_math_kernel_v(
$assignvariableop_60_adam_math_bias_v*
&assignvariableop_61_adam_stat_kernel_v(
$assignvariableop_62_adam_stat_bias_v(
$assignvariableop_63_adam_qb_kernel_v&
"assignvariableop_64_adam_qb_bias_v(
$assignvariableop_65_adam_qf_kernel_v&
"assignvariableop_66_adam_qf_bias_v
identity_68??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?!
value?!B?!DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_cs_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_cs_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_ph_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_ph_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_math_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_math_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_stat_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_stat_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_qb_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_qb_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_qf_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_qf_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_3Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_4Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_4Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_5Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_5Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_6Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_6Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_7Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_7Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_8Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_8Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_9Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_9Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_10Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_10Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_11Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_11Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_12Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_12Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp$assignvariableop_43_adam_cs_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp"assignvariableop_44_adam_cs_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp$assignvariableop_45_adam_ph_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp"assignvariableop_46_adam_ph_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp&assignvariableop_47_adam_math_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_adam_math_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp&assignvariableop_49_adam_stat_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp$assignvariableop_50_adam_stat_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp$assignvariableop_51_adam_qb_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp"assignvariableop_52_adam_qb_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp$assignvariableop_53_adam_qf_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp"assignvariableop_54_adam_qf_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp$assignvariableop_55_adam_cs_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp"assignvariableop_56_adam_cs_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp$assignvariableop_57_adam_ph_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp"assignvariableop_58_adam_ph_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_math_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_math_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp&assignvariableop_61_adam_stat_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp$assignvariableop_62_adam_stat_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp$assignvariableop_63_adam_qb_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp"assignvariableop_64_adam_qb_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp$assignvariableop_65_adam_qf_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp"assignvariableop_66_adam_qf_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67?
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference_loss_fn_2_110238:
6math_kernel_regularizer_square_readvariableop_resource
identity??-MATH/kernel/Regularizer/Square/ReadVariableOp?
-MATH/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6math_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-MATH/kernel/Regularizer/Square/ReadVariableOp?
MATH/kernel/Regularizer/SquareSquare5MATH/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2 
MATH/kernel/Regularizer/Square?
MATH/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MATH/kernel/Regularizer/Const?
MATH/kernel/Regularizer/SumSum"MATH/kernel/Regularizer/Square:y:0&MATH/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/Sum?
MATH/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
MATH/kernel/Regularizer/mul/x?
MATH/kernel/Regularizer/mulMul&MATH/kernel/Regularizer/mul/x:output:0$MATH/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
MATH/kernel/Regularizer/mul?
IdentityIdentityMATH/kernel/Regularizer/mul:z:0.^MATH/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-MATH/kernel/Regularizer/Square/ReadVariableOp-MATH/kernel/Regularizer/Square/ReadVariableOp
?
?
>__inference_QB_layer_call_and_return_conditional_losses_110142

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?+QB/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
+QB/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+QB/kernel/Regularizer/Square/ReadVariableOp?
QB/kernel/Regularizer/SquareSquare3QB/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
QB/kernel/Regularizer/Square?
QB/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
QB/kernel/Regularizer/Const?
QB/kernel/Regularizer/SumSum QB/kernel/Regularizer/Square:y:0$QB/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/Sum
QB/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
QB/kernel/Regularizer/mul/x?
QB/kernel/Regularizer/mulMul$QB/kernel/Regularizer/mul/x:output:0"QB/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
QB/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^QB/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+QB/kernel/Regularizer/Square/ReadVariableOp+QB/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
abstract3
serving_default_abstract:0???????????
9
title0
serving_default_title:0???????????6
CS0
StatefulPartitionedCall:0?????????8
MATH0
StatefulPartitionedCall:1?????????6
PH0
StatefulPartitionedCall:2?????????6
QB0
StatefulPartitionedCall:3?????????6
QF0
StatefulPartitionedCall:4?????????8
STAT0
StatefulPartitionedCall:5?????????tensorflow/serving/predict:??
?Y
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?U
_tf_keras_network?U{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "title"}, "name": "title", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "abstract"}, "name": "abstract", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["title", 0, 0, {}], ["abstract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CS", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "CS", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "PH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "PH", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "MATH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "MATH", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "STAT", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "STAT", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "QB", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "QB", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "QF", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "QF", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}], "input_layers": [["title", 0, 0], ["abstract", 0, 0]], "output_layers": [["CS", 0, 0], ["PH", 0, 0], ["MATH", 0, 0], ["STAT", 0, 0], ["QB", 0, 0], ["QF", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20000]}, {"class_name": "TensorShape", "items": [null, 20000]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "title"}, "name": "title", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "abstract"}, "name": "abstract", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["title", 0, 0, {}], ["abstract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CS", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "CS", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "PH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "PH", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "MATH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "MATH", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "STAT", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "STAT", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "QB", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "QB", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "QF", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "QF", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}], "input_layers": [["title", 0, 0], ["abstract", 0, 0]], "output_layers": [["CS", 0, 0], ["PH", 0, 0], ["MATH", 0, 0], ["STAT", 0, 0], ["QB", 0, 0], ["QF", 0, 0]]}}, "training_config": {"loss": {"class_name": "Custom>BinaryFocalLoss", "config": {"reduction": "auto", "name": null, "gamma": 2.0, "pos_weight": null, "from_logits": false, "label_smoothing": null}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "CS_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "PH_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "MATH_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "STAT_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "QB_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "QF_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "title", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "title"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "abstract", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "abstract"}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20000]}, {"class_name": "TensorShape", "items": [null, 20000]}]}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "CS", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "CS", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "PH", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "PH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "MATH", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MATH", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?	

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "STAT", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "STAT", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?	

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "QB", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "QB", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?	

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "QF", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "QF", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40000}}}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40000]}}
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratem?m?m?m? m?!m?&m?'m?,m?-m?2m?3m?v?v?v?v? v?!v?&v?'v?,v?-v?2v?3v?"
	optimizer
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
v
0
1
2
3
 4
!5
&6
'7
,8
-9
210
311"
trackable_list_wrapper
v
0
1
2
3
 4
!5
&6
'7
,8
-9
210
311"
trackable_list_wrapper
?
=non_trainable_variables

>layers
regularization_losses
trainable_variables
?layer_metrics
@layer_regularization_losses
	variables
Ametrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables

Clayers
regularization_losses
trainable_variables
Dlayer_metrics
Elayer_regularization_losses
	variables
Fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	CS/kernel
:2CS/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
regularization_losses
trainable_variables
Ilayer_metrics
Jlayer_regularization_losses
	variables
Kmetrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	PH/kernel
:2PH/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Lnon_trainable_variables

Mlayers
regularization_losses
trainable_variables
Nlayer_metrics
Olayer_regularization_losses
	variables
Pmetrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2MATH/kernel
:2	MATH/bias
(
?0"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
"regularization_losses
#trainable_variables
Slayer_metrics
Tlayer_regularization_losses
$	variables
Umetrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2STAT/kernel
:2	STAT/bias
(
?0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
(regularization_losses
)trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
*	variables
Zmetrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	QB/kernel
:2QB/bias
(
?0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
[non_trainable_variables

\layers
.regularization_losses
/trainable_variables
]layer_metrics
^layer_regularization_losses
0	variables
_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	QF/kernel
:2QF/bias
(
?0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
`non_trainable_variables

alayers
4regularization_losses
5trainable_variables
blayer_metrics
clayer_regularization_losses
6	variables
dmetrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12"
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
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	rtotal
	scount
t	variables
u	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	vtotal
	wcount
x	variables
y	keras_api"?
_tf_keras_metricp{"class_name": "Mean", "name": "CS_loss", "dtype": "float32", "config": {"name": "CS_loss", "dtype": "float32"}}
?
	ztotal
	{count
|	variables
}	keras_api"?
_tf_keras_metricp{"class_name": "Mean", "name": "PH_loss", "dtype": "float32", "config": {"name": "PH_loss", "dtype": "float32"}}
?
	~total
	count
?	variables
?	keras_api"?
_tf_keras_metrict{"class_name": "Mean", "name": "MATH_loss", "dtype": "float32", "config": {"name": "MATH_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metrict{"class_name": "Mean", "name": "STAT_loss", "dtype": "float32", "config": {"name": "STAT_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricp{"class_name": "Mean", "name": "QB_loss", "dtype": "float32", "config": {"name": "QB_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricp{"class_name": "Mean", "name": "QF_loss", "dtype": "float32", "config": {"name": "QF_loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "CS_accuracy", "dtype": "float32", "config": {"name": "CS_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "PH_accuracy", "dtype": "float32", "config": {"name": "PH_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "MATH_accuracy", "dtype": "float32", "config": {"name": "MATH_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "STAT_accuracy", "dtype": "float32", "config": {"name": "STAT_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "QB_accuracy", "dtype": "float32", "config": {"name": "QB_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "QF_accuracy", "dtype": "float32", "config": {"name": "QF_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
.
r0
s1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
:  (2total
:  (2count
.
v0
w1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
.
z0
{1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
": 
??2Adam/CS/kernel/m
:2Adam/CS/bias/m
": 
??2Adam/PH/kernel/m
:2Adam/PH/bias/m
$:"
??2Adam/MATH/kernel/m
:2Adam/MATH/bias/m
$:"
??2Adam/STAT/kernel/m
:2Adam/STAT/bias/m
": 
??2Adam/QB/kernel/m
:2Adam/QB/bias/m
": 
??2Adam/QF/kernel/m
:2Adam/QF/bias/m
": 
??2Adam/CS/kernel/v
:2Adam/CS/bias/v
": 
??2Adam/PH/kernel/v
:2Adam/PH/bias/v
$:"
??2Adam/MATH/kernel/v
:2Adam/MATH/bias/v
$:"
??2Adam/STAT/kernel/v
:2Adam/STAT/bias/v
": 
??2Adam/QB/kernel/v
:2Adam/QB/bias/v
": 
??2Adam/QF/kernel/v
:2Adam/QF/bias/v
?2?
&__inference_model_layer_call_fn_109243
&__inference_model_layer_call_fn_109934
&__inference_model_layer_call_fn_109888
&__inference_model_layer_call_fn_109420?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_109842
A__inference_model_layer_call_and_return_conditional_losses_108934
A__inference_model_layer_call_and_return_conditional_losses_109065
A__inference_model_layer_call_and_return_conditional_losses_109674?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_108469?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *Q?N
L?I
!?
title???????????
$?!
abstract???????????
?2?
,__inference_concatenate_layer_call_fn_109947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_109941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_CS_layer_call_fn_109979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_CS_layer_call_and_return_all_conditional_losses_109990?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_PH_layer_call_fn_110022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_PH_layer_call_and_return_all_conditional_losses_110033?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_MATH_layer_call_fn_110065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_MATH_layer_call_and_return_all_conditional_losses_110076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_STAT_layer_call_fn_110108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_STAT_layer_call_and_return_all_conditional_losses_110119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_QB_layer_call_fn_110151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_QB_layer_call_and_return_all_conditional_losses_110162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_QF_layer_call_fn_110194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_QF_layer_call_and_return_all_conditional_losses_110205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_110216?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_110227?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_110238?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_110249?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_110260?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_110271?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
$__inference_signature_wrapper_109506abstracttitle"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_CS_activity_regularizer_108482?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
>__inference_CS_layer_call_and_return_conditional_losses_109970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_PH_activity_regularizer_108495?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
>__inference_PH_layer_call_and_return_conditional_losses_110013?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_MATH_activity_regularizer_108508?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
@__inference_MATH_layer_call_and_return_conditional_losses_110056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_STAT_activity_regularizer_108521?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
@__inference_STAT_layer_call_and_return_conditional_losses_110099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_QB_activity_regularizer_108534?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
>__inference_QB_layer_call_and_return_conditional_losses_110142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_QF_activity_regularizer_108547?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
>__inference_QF_layer_call_and_return_conditional_losses_110185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 W
*__inference_CS_activity_regularizer_108482)?
?
?
self
? "? ?
B__inference_CS_layer_call_and_return_all_conditional_losses_109990l1?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
>__inference_CS_layer_call_and_return_conditional_losses_109970^1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? x
#__inference_CS_layer_call_fn_109979Q1?.
'?$
"?
inputs???????????
? "??????????Y
,__inference_MATH_activity_regularizer_108508)?
?
?
self
? "? ?
D__inference_MATH_layer_call_and_return_all_conditional_losses_110076l !1?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
@__inference_MATH_layer_call_and_return_conditional_losses_110056^ !1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? z
%__inference_MATH_layer_call_fn_110065Q !1?.
'?$
"?
inputs???????????
? "??????????W
*__inference_PH_activity_regularizer_108495)?
?
?
self
? "? ?
B__inference_PH_layer_call_and_return_all_conditional_losses_110033l1?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
>__inference_PH_layer_call_and_return_conditional_losses_110013^1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? x
#__inference_PH_layer_call_fn_110022Q1?.
'?$
"?
inputs???????????
? "??????????W
*__inference_QB_activity_regularizer_108534)?
?
?
self
? "? ?
B__inference_QB_layer_call_and_return_all_conditional_losses_110162l,-1?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
>__inference_QB_layer_call_and_return_conditional_losses_110142^,-1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? x
#__inference_QB_layer_call_fn_110151Q,-1?.
'?$
"?
inputs???????????
? "??????????W
*__inference_QF_activity_regularizer_108547)?
?
?
self
? "? ?
B__inference_QF_layer_call_and_return_all_conditional_losses_110205l231?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
>__inference_QF_layer_call_and_return_conditional_losses_110185^231?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? x
#__inference_QF_layer_call_fn_110194Q231?.
'?$
"?
inputs???????????
? "??????????Y
,__inference_STAT_activity_regularizer_108521)?
?
?
self
? "? ?
D__inference_STAT_layer_call_and_return_all_conditional_losses_110119l&'1?.
'?$
"?
inputs???????????
? "3?0
?
0?????????
?
?	
1/0 ?
@__inference_STAT_layer_call_and_return_conditional_losses_110099^&'1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? z
%__inference_STAT_layer_call_fn_110108Q&'1?.
'?$
"?
inputs???????????
? "???????????
!__inference__wrapped_model_108469?23,-&' ![?X
Q?N
L?I
!?
title???????????
$?!
abstract???????????
? "???
"
CS?
CS?????????
&
MATH?
MATH?????????
"
PH?
PH?????????
"
QB?
QB?????????
"
QF?
QF?????????
&
STAT?
STAT??????????
G__inference_concatenate_layer_call_and_return_conditional_losses_109941?^?[
T?Q
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
? "'?$
?
0???????????
? ?
,__inference_concatenate_layer_call_fn_109947|^?[
T?Q
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
? "????????????;
__inference_loss_fn_0_110216?

? 
? "? ;
__inference_loss_fn_1_110227?

? 
? "? ;
__inference_loss_fn_2_110238 ?

? 
? "? ;
__inference_loss_fn_3_110249&?

? 
? "? ;
__inference_loss_fn_4_110260,?

? 
? "? ;
__inference_loss_fn_5_1102712?

? 
? "? ?
A__inference_model_layer_call_and_return_conditional_losses_108934?23,-&' !c?`
Y?V
L?I
!?
title???????????
$?!
abstract???????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
W?T
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 
?	
1/5 ?
A__inference_model_layer_call_and_return_conditional_losses_109065?23,-&' !c?`
Y?V
L?I
!?
title???????????
$?!
abstract???????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
W?T
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 
?	
1/5 ?
A__inference_model_layer_call_and_return_conditional_losses_109674?23,-&' !f?c
\?Y
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
W?T
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 
?	
1/5 ?
A__inference_model_layer_call_and_return_conditional_losses_109842?23,-&' !f?c
\?Y
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
W?T
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 
?	
1/5 ?
&__inference_model_layer_call_fn_109243?23,-&' !c?`
Y?V
L?I
!?
title???????????
$?!
abstract???????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
&__inference_model_layer_call_fn_109420?23,-&' !c?`
Y?V
L?I
!?
title???????????
$?!
abstract???????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
&__inference_model_layer_call_fn_109888?23,-&' !f?c
\?Y
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
&__inference_model_layer_call_fn_109934?23,-&' !f?c
\?Y
O?L
$?!
inputs/0???????????
$?!
inputs/1???????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5??????????
$__inference_signature_wrapper_109506?23,-&' !k?h
? 
a?^
0
abstract$?!
abstract???????????
*
title!?
title???????????"???
"
CS?
CS?????????
&
MATH?
MATH?????????
"
PH?
PH?????????
"
QB?
QB?????????
"
QF?
QF?????????
&
STAT?
STAT?????????