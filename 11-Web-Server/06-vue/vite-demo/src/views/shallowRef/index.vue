<template>
  <div>
    <h1>shallowRef & triggerRef</h1>
    <h3>{{ arr1 }}</h3>
    <h3>{{ arr2 }}</h3>
    <h3>{{ arr3 }}</h3>
    <button @click="arr3.push(4)">arr3 push</button>
    <button @click="arr3 = [4, 5, 6]">修改arr3</button>
    <button @click="trigger">triggerRef 主动触发派发更新</button>
  </div>
</template>

<script setup>
import {reactive, ref, shallowRef, triggerRef} from 'vue'

//reactive 清空数组 & 赋值数据
let arr1 = reactive([1, 2, 3])
arr1.splice(0, 3)
arr1.push(4, 5, 6)

//ref 清空数组 & 赋值数组
//ref 传入引用类型数据 会被reactive解包
const arr2 = ref([1, 2, 3])
arr2.value = []
arr2.value = [4, 5, 6]
console.log('arr2', arr2)

//shallowRef 浅ref 修改value是响应式 修改value值里面的属性不是响应式
const arr3 = shallowRef([1, 2, 3])
console.log('arr3', arr3)

//主动派发更新 让shallowRef修改的深层数据 也渲染视图
function trigger() {
  arr3.value.push(4)
  console.log('arr3', arr3)
  //手动触发arr3 更新
  triggerRef(arr3)
}
</script>

<style lang="scss" scoped></style>