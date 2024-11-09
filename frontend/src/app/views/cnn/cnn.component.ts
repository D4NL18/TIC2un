import { Component, ViewChild } from '@angular/core';
import { CnnResultsComponent } from '../../components/cnn-results/cnn-results.component';
import { ButtonComponent } from '../../components/button/button.component';
import { NavComponent } from '../../components/nav/nav.component';

@Component({
  selector: 'app-cnn',
  standalone: true,
  imports: [CnnResultsComponent, ButtonComponent, NavComponent],
  templateUrl: './cnn.component.html',
  styleUrl: './cnn.component.scss'
})
export class CnnComponent {
  @ViewChild(CnnResultsComponent) cnnResultsComponent!: CnnResultsComponent

  onTrainCNN() {
    if(this.cnnResultsComponent) {
      this.cnnResultsComponent.trainCNN_FT()
      this.cnnResultsComponent.trainCNN_TF()
    }
  }
}
