import { Routes } from '@angular/router';
import { SomComponent } from './views/som/som.component';
import { SvmComponent } from './views/svm/svm.component';

export const routes: Routes = [
    {
        path: "som",
        component: SomComponent
    },
    {
        path: "",
        component: SvmComponent
    },
];
