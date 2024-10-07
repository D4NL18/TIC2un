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

  imageUrl: string | undefined;
  isLoading: boolean = false;

  constructor(private somService: SomService) { }

  trainSom() {
    console.log("Treinando SOM...");
    this.isLoading = true;
    this.somService.trainSom().subscribe({
      next: (response) => {
        console.log('Treinamento concluído', response);
        this.fetchImage();
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
    this.somService.getImage().subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        this.imageUrl = url;
      },
      error: (err) => {
        console.error('Erro ao buscar imagem:', err);
      }
    });
  }
}