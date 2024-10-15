import { Component } from '@angular/core';
import { DeeplearningResultsComponent } from '../../components/deeplearning-results/deeplearning-results.component';
import { ButtonComponent } from '../../components/button/button.component';
import { NavComponent } from '../../components/nav/nav.component';

@Component({
  selector: 'app-deeplearning',
  standalone: true,
  imports: [DeeplearningResultsComponent, ButtonComponent, NavComponent],
  templateUrl: './deeplearning.component.html',
  styleUrl: './deeplearning.component.scss'
})
export class DeeplearningComponent {
  onTrainDeepLearning() {

  }
}
