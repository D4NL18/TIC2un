import { Routes } from '@angular/router';
import { SomComponent } from './views/som/som.component';
import { SvmComponent } from './views/svm/svm.component';
import { DeeplearningComponent } from './views/deeplearning/deeplearning.component';

export const routes: Routes = [
    {
        path: "som",
        component: SomComponent
    },
    {
        path: "deeplearning",
        component: DeeplearningComponent
    },
    {
        path: "",
        component: SvmComponent
    },
];
