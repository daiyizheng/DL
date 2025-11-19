<template>
  <div>
    <h1>ref 基本响应式数据</h1>
    <h3>简单数据:{{ num1 }}</h3>
    <button @click="addNum1">num1++</button>
    <hr />
    <h3>响应式数据:{{ num2 }}</h3>
    <button @click="addNum2">num2++</button>
    <!-- 模板中自动解包ref的value属性 -->
    <button @click="num2++">num2+1</button>

    <hr />
    <h3>ref获取dom元素</h3>
    <div ref="divRef">我是div</div>
  </div>
</template>

<script setup>
//onMounted 相当于vue2 mounted生命周期 挂载后
import {ref, onMounted} from 'vue'
//简单数据 并不是响应式数据
let num1 = 10
function addNum1() {
  num1++
  console.log('num1', num1)
}

//响应式数据 当数据变化 会自动渲染视图
const num2 = ref(20)
// console.log('num2', num2)
//在js中修改数据 需要修改.value属性
function addNum2() {
  num2.value++
}

//vue3 响应式数据基本类型 ref
//通过class创建一个类 里面get value 做依赖收集
//get value 做派发更新

class MyRef {
  constructor(value) {
    this._value = value
  }
  //访问器
  get value() {
    console.log('触发getter函数 访问')
    return this._value
  }
  //读取器
  set value(newVal) {
    console.log('触发setter函数 修改')
    this._value = newVal
  }
}

const c1 = new MyRef(20)
console.log('c1', c1)
//访问属性
console.log(c1.value)
//修改属性
c1.value = 30
console.log(c1.value)

//ref获取dom
const divRef = ref(null)
//挂载后
onMounted(() => {
  console.log(111, divRef.value)
})
</script>

<style lang="scss" scoped></style>