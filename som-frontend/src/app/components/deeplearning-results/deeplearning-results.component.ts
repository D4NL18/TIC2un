import { Component, EventEmitter, Output } from '@angular/core';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';
import { CommonModule } from '@angular/common';
import { DeeplearningService } from '../../services/deeplearning.service';

@Component({
  selector: 'app-deeplearning-results',
  standalone: true,
  imports: [LoadingSpinnerComponent, CommonModule],
  templateUrl: './deeplearning-results.component.html',
  styleUrl: './deeplearning-results.component.scss'
})
export class DeeplearningResultsComponent {
  @Output() trainRequested = new EventEmitter<void>()

  imageUrlPT: string | undefined
  imageUrlTF: string | undefined
  accuracyPT: number | undefined
  accuracyTF: number | undefined
  isLoading: boolean = false;

  constructor(private dlService: DeeplearningService) { }

  trainDeepLearning() {
    console.log("Treinando DL...")
    this.isLoading = true
    this.dlService.trainDeepLearning().subscribe({
      next: (response) => {
        console.log("Treinamento Concluido", response)
        this.fetchImage()
        this.fetchAccuracy()
      },
      error: (err) => {
        console.log("Erro ao treinar DL: ", err)
      },
      complete: () => {
        this.isLoading = false
      }
    })
  }

  fetchImage() {
    this.dlService.getImage("pt").subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrlPT = url
      },
      error: (err) => {
        console.error('Erro ao buscar imagem PT:', err);
      }
    })
    this.dlService.getImage("tf").subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrlTF = url
      },
      error: (err) => {
        console.error('Erro ao buscar imagem TF:', err);
      }
    })
  }

  fetchAccuracy() {
    this.dlService.getAccuracy('pt').subscribe({
      next: (response) => {
        this.accuracyPT = response.accuracy
      },
      error: (err) => {
        console.error('Erro ao buscar precisão PT:', err);
      }
    })
    this.dlService.getAccuracy('tf').subscribe({
      next: (response) => {
        this.accuracyTF = response.accuracy
      },
      error: (err) => {
        console.error('Erro ao buscar precisão TF:', err);
      }
    })
  }

}
