import { Component } from '@angular/core';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';

@Component({
  selector: 'app-deeplearning-results',
  standalone: true,
  imports: [LoadingSpinnerComponent],
  templateUrl: './deeplearning-results.component.html',
  styleUrl: './deeplearning-results.component.scss'
})
export class DeeplearningResultsComponent {

}
