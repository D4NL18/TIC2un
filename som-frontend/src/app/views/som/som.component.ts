import { Component } from '@angular/core';
import { NavComponent } from '../../components/nav/nav.component';
import { ButtonComponent } from '../../components/button/button.component';
import { SomResultsComponent } from '../../components/som-results/som-results.component';

@Component({
  selector: 'app-som',
  standalone: true,
  imports: [NavComponent, ButtonComponent, SomResultsComponent],
  templateUrl: './som.component.html',
  styleUrl: './som.component.scss'
})
export class SomComponent {

}
