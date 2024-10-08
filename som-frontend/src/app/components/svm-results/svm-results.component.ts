import { Component, EventEmitter, Output } from '@angular/core';
import { SvmService } from '../../services/svm.service';
import { CommonModule } from '@angular/common';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';

@Component({
  selector: 'app-svm-results',
  standalone: true,
  imports: [CommonModule, LoadingSpinnerComponent],
  templateUrl: './svm-results.component.html',
  styleUrls: ['./svm-results.component.scss']
})
export class SvmResultsComponent {

  @Output() executionRequested = new EventEmitter<void>();

  imageUrl: string | undefined;
  accuracy: number | undefined;
  isLoading: boolean = false;

  constructor(private svmService: SvmService) {}

  runSVM() {
    console.log("Executando SVM...");
    this.isLoading = true;
    this.svmService.runSVM().subscribe({
      next: (response) => {
        console.log("Treinamento Concluído", response);
        this.fetchResults();
      },
      error: (err) => {
        console.log('Erro ao executar SVM', err);
      },
      complete: () => {
        this.isLoading = false;
      }
    });
  }

  fetchResults() {
    this.svmService.getResults().subscribe({
      next: (response) => {
        this.accuracy = response.accuracy;
        this.imageUrl = response.image_url;
      },
      error: (err) => {
        console.log('Erro ao encontrar resultados', err);
      }
    });
  }
}
