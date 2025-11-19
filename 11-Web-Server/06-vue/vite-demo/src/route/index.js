import { createRouter, createWebHashHistory } from "vue-router";

import Layout from "@/layout/layout.vue";

export const routes = [
  {
    path: "/",
    redirect: "/base",
  },

  {
    path: "/base",
    component: Layout,
    children: [
      {
        path: "",
        component: () => import("@/views/base/index.vue"),
      },
    ],
  },
  {
    path: "/base-directive",
    component: Layout,
    children: [
      {
        path: "",
        component: () => import("@/views/base-directive/index.vue"),
      },
    ],
  },
  {
    path: "/css-scope",
    component: Layout,
    children: [
      {
        path: "",
        component: () => import("@/views/css-scope/index.vue"),
      },
    ],
  },
  {
    path: '/setup', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/setup/index.vue'),
      },
    ],
  },
  {
    path: '/ref', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/ref/index.vue'),
      },
    ],
  },
  {
    path: '/reactive', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/reactive/index.vue'),
      },
    ],
  },
  {
    path: '/toRef-toRefs', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/toRef-toRefs/index.vue'),
      },
    ],
  },
  {
    path: '/shallowRef', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/shallowRef/index.vue'),
      },
    ],
  },
  {
    path: '/customRef', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/customRef/index.vue'),
      },
    ],
  },
  {
    path: '/utils-fn', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/utils-fn/index.vue'),
      },
    ],
  },
  {
    path: '/computed', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/computed/index.vue'),
      },
    ],
  },
  {
    path: '/watch', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/watch/index.vue'),
      },
    ],
  },
  {
    path: '/watchEffect', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/watchEffect/index.vue'),
      },
    ],
  },
  {
    path: '/life-cycle', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/life-cycle/index.vue'),
      },
    ],
  },
  {
    path: '/father-child', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/father-child/index.vue'),
      },
    ],
  },
  {
    path: '/v-model', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/v-model/index.vue'),
      },
    ],
  },
  {
    path: '/tabs', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/tabs/index.vue'),
      },
    ],
  },
  {
    path: '/shopcar', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/shopcar/index.vue'),
      },
    ],
  },
  {
    path: '/todo', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/todo/index.vue'),
      },
    ],
  },
  {
    path: '/slot', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/slot/index.vue'),
      },
    ],
  },
  {
    path: '/expose', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/expose/index.vue'),
      },
    ],
  },
  {
    path: '/attrs', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/attrs/index.vue'),
      },
    ],
  },
  {
    path: '/com', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/com/index.vue'),
      },
    ],
  },
  {
    path: '/provider', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/provider/index.vue'),
      },
    ],
  },
  {
    path: '/mitt', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/mitt/index.vue'),
      },
    ],
  },
  {
    path: '/hooks', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/hooks/index.vue'),
      },
    ],
  },
  {
    path: '/directive', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/directive/index.vue'),
      },
    ],
  },
  {
    path: '/plugin', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/plugin/index.vue'),
      },
    ],
  },
  {
    path: '/transition', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/transition/index.vue'),
      },
    ],
  },
  {
    path: '/keep-alive', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/keep-alive/index.vue'),
      },
    ],
  },
  {
    path: '/teleport', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/teleport/index.vue'),
      },
    ],
  },
  {
    path: '/suspense', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        name: 'suspense',
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/suspense/index.vue'),
      },
    ],
  },
  {
    path: '/recursion', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/recursion/index.vue'),
      },
    ],
  },
  {
    path: '/user/:name',
    name: 'user',
    component: () => import('@/views/user/index.vue'),
  },
  {
    path: '/pinia-1', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/pinia-1/index.vue'),
      },
    ],
  },
  {
    path: '/pinia-2', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/pinia-2/index.vue'),
      },
    ],
  },
  {
    path: '/render', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/render/index.vue'),
      },
    ],
  },
  {
    path: '/jsx-tem', //配置路由地址
    component: Layout, //配置组件
    children: [
      //children里面渲染二级路由
      {
        path: '',
        //路由懒加载 只要输入路由地址才会加载
        component: () => import('@/views/jsx-tem/index.jsx'),
      },
    ],
  },
];

const router = createRouter({
  routes,
  history: createWebHashHistory(),
});

export default router;
