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
];

const router = createRouter({
  routes,
  history: createWebHashHistory(),
});

export default router;
