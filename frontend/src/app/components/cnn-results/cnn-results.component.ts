import { Component, EventEmitter, Output } from '@angular/core';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';
import { CommonModule } from '@angular/common';
import { CnnService } from '../../services/cnn.service';

@Component({
  selector: 'app-cnn-results',
  standalone: true,
  imports: [LoadingSpinnerComponent, CommonModule],
  templateUrl: './cnn-results.component.html',
  styleUrl: './cnn-results.component.scss'
})
export class CnnResultsComponent {
  @Output() trainRequested = new EventEmitter<void>()

  imageUrl: string | undefined
  accuracy: number | undefined
  f1_score: number | undefined
  isLoading: boolean = false;

  constructor(private cnnService: CnnService) {}

  trainCNN() {
    console.log("Treinando CNN...")
    this.isLoading = true
    this.cnnService.trainCNN().subscribe({
      next: (response) => {
        console.log("Treinamento concluído", response)
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
    this.cnnService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrl = url
      },
      error: (err) => {
        console.log("Erro ao buscar imagem CNN: ", err)
      }
    })
  }

  fetchAccuracy() {
    this.cnnService.getAccuracy().subscribe({
      next: (response) => {
        this.accuracy = response.accuracy
        this.f1_score = response.f1_score
      },
      error: (err) => {
        console.log("Erro ao buscar metricas: ", err)
      }
    }) 
  }

}
