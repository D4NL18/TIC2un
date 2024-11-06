import { Component, EventEmitter, Output } from '@angular/core';
import { SomService } from '../../services/som.service';
import { CommonModule } from '@angular/common';
import { LoadingSpinnerComponent } from '../loading-spinner/loading-spinner.component';

@Component({
  selector: 'app-som-results',
  standalone: true,
  templateUrl: './som-results.component.html',
  styleUrls: ['./som-results.component.scss'],
  imports: [CommonModule, LoadingSpinnerComponent]
})
export class SomResultsComponent {
  @Output() trainRequested = new EventEmitter<void>();

  imageUrlManual: string | undefined;
  imageUrlMinisom: string | undefined;
  accuracyManual: number | undefined;
  accuracyMinisom: number | undefined;
  isLoading: boolean = false;

  constructor(private somService: SomService) { }

  trainSom() {
    console.log("Treinando SOM...");
    this.isLoading = true;
    this.somService.trainSom().subscribe({
      next: (response) => {
        console.log('Treinamento concluído', response);
        this.fetchImage();
        this.fetchAccuracy();
      },
      error: (err) => {
        console.error('Erro ao treinar SOM:', err);
      },
      complete: () => {
        this.isLoading = false;
      }
    });
  }

  fetchImage() {
    this.somService.getImage("manual").subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlManual = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem:', err);
      }
    });
    this.somService.getImage("minisom").subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrlMinisom = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem:', err);
      }
    });
  }

  fetchAccuracy() {
    this.somService.getAccuracy('manual').subscribe({
      next: (response) => {
        this.accuracyManual = response.accuracy;
      },
      error: (err) => {
        console.error('Erro ao buscar precisão do SOM manual:', err);
      }
    });

    this.somService.getAccuracy('minisom').subscribe({
      next: (response) => {
        this.accuracyMinisom = response.accuracy;
      },
      error: (err) => {
        console.error('Erro ao buscar precisão do MiniSom:', err);
      }
    });
  }
}
