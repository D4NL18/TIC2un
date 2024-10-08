import { Component, EventEmitter, Output } from '@angular/core';

import { SvmService } from '../../services/svm.service';
import { CommonModule } from '@angular/common';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';

@Component({
  selector: 'app-svm-results',
  standalone: true,
  imports: [CommonModule, LoadingSpinnerComponent],
  templateUrl: './svm-results.component.html',
  styleUrl: './svm-results.component.scss'
})
export class SvmResultsComponent {

  @Output() executionRequested = new EventEmitter<void>()

  imageUrl: string | undefined
  accuracy: number | undefined
  isLoading: boolean = false

  constructor(private svmService: SvmService) {}

  runSVM() {
    console.log("Executando SVM...")
    this.isLoading = true
    this.svmService.runSVM().subscribe({
      next: (response) => {
        console.log("Treinamento Concluído", response)
        this.accuracy = response.accuracy
        this.fetchImage();
      },
      error: (err) => {
        console.log('Erro ao executar SVM ', err)
      },
      complete: () => {
        this.isLoading = false
      }
    })
  }
  fetchImage() {
    this.svmService.getResults().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob)
        this.imageUrl = url
      },
      error: (err) => {
        console.log('Erro ao encontrar resultados', err)
      }
    })
  }

}
