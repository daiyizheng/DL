<template>
  <div>
    <h1>基本语法</h1>
    <p>我是p标签</p>
    <h3>胡须语法</h3>

    <h3>变量:{{ num1 }}</h3>
    <h3>函数:{{ getMsg() }}</h3>
    <h3>四则运算:{{ num1 + num2 }}</h3>
    <h3>三目运算:{{ false ? 40 : 60 }}</h3>
    <h3>短路运算符:{{ true && '你好啊' }}</h3>
    <h3>正则表达式:{{ /www/.test('wwwwww') }}</h3>
    <hr />
    <h3>{{ msg }}</h3>
  </div>
</template>

<script>
export default {
  setup() {
    //里面写组合式api代码  需要被模板使用的方法和数据 return 出来

    //胡须语法 {{表达式}}
    const num1 = 20

    //函数
    function getMsg() {
      return '你好啊'
    }

    //四则运算
    const num2 = 30

    const msg = '你好啊'

    //暴露出去的数据和方法 能被模板使用
    return {
      num1,
      num2,
      getMsg,
      msg,
    }
  },
}
</script>

<style lang="scss" scoped></style>