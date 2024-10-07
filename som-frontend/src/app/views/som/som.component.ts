import { Component, ViewChild } from '@angular/core';
import { NavComponent } from '../../components/nav/nav.component';
import { ButtonComponent } from '../../components/button/button.component';
import { SomResultsComponent } from '../../components/som-results/som-results.component';

@Component({
  selector: 'app-som',
  standalone: true,
  imports: [NavComponent, ButtonComponent, SomResultsComponent],
  templateUrl: './som.component.html',
  styleUrls: ['./som.component.scss'] // Corrigido para 'styleUrls' em vez de 'styleUrl'
})
export class SomComponent {
  @ViewChild(SomResultsComponent) somResultsComponent!: SomResultsComponent; // Adiciona uma referência ao componente filho

  onTrainSom() {
    if (this.somResultsComponent) { // Verifica se a referência está definida
      this.somResultsComponent.trainSom(); // Chama a função trainSom do componente filho
    }
  }
}
