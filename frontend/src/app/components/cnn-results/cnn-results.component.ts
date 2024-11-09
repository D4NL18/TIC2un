import { Component, EventEmitter, Output } from '@angular/core';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';
import { CommonModule } from '@angular/common';
import { CnnService } from '../../services/cnn.service';
import { CnnFinetunningService } from '../../services/cnn-finetunning.service';

@Component({
  selector: 'app-cnn-results',
  standalone: true,
  imports: [LoadingSpinnerComponent, CommonModule],
  templateUrl: './cnn-results.component.html',
  styleUrl: './cnn-results.component.scss'
})
export class CnnResultsComponent {
  @Output() trainRequested = new EventEmitter<void>()

  imageUrlTF: string | undefined
  accuracyTF: number | undefined
  f1_scoreTF: number | undefined
  imageUrlFT: string | undefined
  accuracyFT: number | undefined
  f1_scoreFT: number | undefined
  isLoading: boolean = false;

  constructor(private cnnService: CnnService, private cnn_ftService: CnnFinetunningService) {}

  trainCNN_TF() {
    console.log("Treinando CNN TF...")
    this.isLoading = true
    this.cnnService.trainCNN().subscribe({
      next: (response) => {
        console.log("Treinamento TF concluído", response)
        this.fetchImageTF()
        this.fetchAccuracyTF()
      },
      error: (err) => {
        console.log("Erro ao treinar TF: ", err)
      },
      complete: () => {
        this.isLoading = false
      }
    })
  }
  trainCNN_FT() {
    console.log("Treinando CNN FT...")
    this.isLoading = true
    this.cnn_ftService.trainCNN().subscribe({
      next: (response) => {
        console.log("Treinamento FT concluído", response)
        this.fetchImageFT()
        this.fetchAccuracyFT()
      },
      error: (err) => {
        console.log("Erro ao treinar CNN FT: ", err)
      },
      complete: () => {
        this.isLoading = false
      }
    })
  }

  fetchImageTF() {
    this.cnnService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrlTF = url
      },
      error: (err) => {
        console.log("Erro ao buscar imagem CNN TF: ", err)
      }
    })
  }
  fetchImageFT() {
    this.cnn_ftService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrlFT = url
      },
      error: (err) => {
        console.log("Erro ao buscar imagem CNN FT: ", err)
      }
    })
  }

  fetchAccuracyTF() {
    this.cnnService.getAccuracy().subscribe({
      next: (response) => {
        this.accuracyTF = response.accuracy
        this.f1_scoreTF = response.f1_score
      },
      error: (err) => {
        console.log("Erro ao buscar metricas TF: ", err)
      }
    }) 
  }
  fetchAccuracyFT() {
    this.cnn_ftService.getAccuracy().subscribe({
      next: (response) => {
        this.accuracyFT = response.accuracy
        this.f1_scoreFT = response.f1_score
      },
      error: (err) => {
        console.log("Erro ao buscar metricas FT: ", err)
      }
    }) 
  }

}
