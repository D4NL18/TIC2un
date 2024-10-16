import { Routes } from '@angular/router';
import { SomComponent } from './views/som/som.component';
import { SvmComponent } from './views/svm/svm.component';
import { DeeplearningComponent } from './views/deeplearning/deeplearning.component';
import { CnnComponent } from './views/cnn/cnn.component';

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
        path: "cnn",
        component: CnnComponent
    },
    {
        path: "",
        component: SvmComponent
    },
];
