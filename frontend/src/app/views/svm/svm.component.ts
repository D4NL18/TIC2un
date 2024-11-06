import { Component, ViewChild } from '@angular/core';
import { SvmResultsComponent } from '../../components/svm-results/svm-results.component';
import { ButtonComponent } from '../../components/button/button.component';
import { NavComponent } from '../../components/nav/nav.component';

@Component({
  selector: 'app-svm',
  standalone: true,
  imports: [SvmResultsComponent, ButtonComponent, NavComponent],
  templateUrl: './svm.component.html',
  styleUrl: './svm.component.scss'
})
export class SvmComponent {
  @ViewChild(SvmResultsComponent) svmResultsComponent!: SvmResultsComponent
  
  onRunSVM() {
    if (this.svmResultsComponent) {
      this.svmResultsComponent.runSVM()
    }
  }
}
